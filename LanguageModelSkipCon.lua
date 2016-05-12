require 'torch'
require 'nn'

require 'VanillaRNN'
require 'LSTM'
require 'dimPrint'		-- Self-written debugging module whose purpose is to print the dimensions of tensors passing through.

local utils = require 'util.utils'


local LM, parent = torch.class('nn.LanguageModelSkipCon', 'nn.Module')


function LM:__init(kwargs)
  self.idx_to_token = utils.get_kwarg(kwargs, 'idx_to_token')
  self.token_to_idx = {}
  self.vocab_size = 0
  for idx, token in pairs(self.idx_to_token) do
    self.token_to_idx[token] = idx
    self.vocab_size = self.vocab_size + 1
  end

  self.model_type = utils.get_kwarg(kwargs, 'model_type')
  self.wordvec_dim = utils.get_kwarg(kwargs, 'wordvec_size')
  self.rnn_size = utils.get_kwarg(kwargs, 'rnn_size')
  self.num_layers = utils.get_kwarg(kwargs, 'num_layers')
  self.dropout = utils.get_kwarg(kwargs, 'dropout')
  self.batchnorm = utils.get_kwarg(kwargs, 'batchnorm')

  local V, D, H = self.vocab_size, self.wordvec_dim, self.rnn_size
	
	--[[ Building Skip Connected network --]]
	--[[ Structure after nn.LookupTable(V, D):
	--	First layer, composed of 2 sub-layers:
	--	l1 is nn.Sequential(), l1[1] is first layer of l1, l1[1][1] is first module of first layer of l1
	--	l1[1][1] = LSTM						
	--	l1[1][2] = nn.Identity()	(For incoming skip connections)
	--	Connections:
	--		l1 -> l1[1][1]					Copying the input to both cells of the first sub-layer
	--		l1 -> l1[1][2]
	--	------------------------
	--	l1[2][1] = nn.Identity()	(For outgoing skip connections - will turn into a join in future layers)
	--	l1[2][2] = nn.Join()			(Join input and previous LSTM output for incoming skip connections)
	--	l1[2][3] = nn.Identity()	(For incoming skip connections, transferring forwards)
	--	Connections:
	--		l1[1][1] -> l1[2][1]		(LSTM -> Outgoing skip 'accumulator' (line of repeated table joins))
	--		
	--		l1[1][1] -> l1[2][2]		(LSTM -> Join with network input (for incoming skip connctions))
	--		l1[1][2] -> l1[2][2]		(Input forwarder -> Join with LSTM (for incoming skip connctions))
	--		
	--		l1[1][2] -> l1[2][3]		(network input forwarder -> network input forwarder))
	--	------------------------
	--	Layers n != 1 are identical to layer 1, EXCEPT:
	--	ln[1][3] = nn.Identity()	(For forwarding network input)
	--	ln[2][3] = nn.Join()			(Outgoing skip 'accumulator' (line of repeated table joins with LSTM outputs))
	--	Additional connections:
	--		ln[1][1] -> ln[2][3]		(LSTM -> outgoing skip accumulator)
	--		ln[1][3] -> ln[2][1]		(network input forwarder -> network input forwarder)
	--]]

  self.net = nn.Sequential()
  self.rnns = {}
  self.bn_view_in = {}
  self.bn_view_out = {}

  self.net:add(nn.LookupTable(V, D))
 	
	for i = 1, self.num_layers do
    -- Selecting input dimensions for LSTM cells
		local prev_dim = D+H								-- All LSTMs in layers 2 onwards have input dimension D+H
    if i == 1 then prev_dim = D end			-- First layer LSTM has input dimension D only
   	
		-- Selecting cell type to use
		local rnn
    if self.model_type == 'rnn' then
      rnn = nn.VanillaRNN(prev_dim, H)
    elseif self.model_type == 'lstm' then
      rnn = nn.LSTM(prev_dim, H)
    end
    rnn.remember_states = true
    table.insert(self.rnns, rnn)				-- This table is used to find all the rnn cells and reset them later.
    
		-- Construct and link the layers

		--self.net:add(rnn)
		
		if i == 1 then	-- Set up first layer
			local t1 = nn.Sequential()			-- Contains both sub-layers
			local t11 = nn.ConcatTable()		-- First sub-layer
			local t12 = nn.ConcatTable()		-- Second sub-layer

			t11:add(rnn)
			t11:add(nn.Identity())		-- output 1 of this layer is LSTM output, output 2 is input.
				-- Output from t11: table of 2 elements:
				-- 	{LSTM output, network input}
			t12:add(nn.SelectTable(1))		-- Grab LSTM output only from first sublayer.
			t12:add(nn.JoinTable(3, 3))			-- For incoming skip connection to next LSTM layer.
			--t12:add(nn.dimPrint("t12 JoinTable Output"))
			t12:add(nn.SelectTable(2))		-- Forward input for use in future incoming skip connections.
				-- Output from t12: table of 3 elements:
				-- 	{LSTM output, LSTM output + network input, network input}
			t1:add(t11)
			--t1:add(nn.dimPrint("output of t11"))
			t1:add(t12)		-- Construct the complete first layer.
			--t1:add(nn.dimPrint("output of t12"))
			self.net:add(t1)	-- Add the completed layer to the overall network container.

		elseif i ~= 1 then	-- Set up any layers after the first layer			
			local t1 = nn.Sequential()			-- Container for both sub-layers
			local t11 = nn.ParallelTable()	-- First sub-layer
			local t12 = nn.ConcatTable()		-- Second sub-layer
			
			t11:add(nn.Identity())					--Outgoing skip connection 'accumulator' forwarder
			t11:add(rnn)
			t11:add(nn.Identity())					-- network input forwarder, for incoming skip connections
				-- Output from t11: table of 3 elements:
				-- 	{outgoing skipcon forwarder, LSTM output, input forwarder}
			local t12s = nn.Sequential()		-- Container for handling outgoing skip connection 
			t12s:add(nn.NarrowTable(1, 2))	-- Select only the first two outputs from t11
			t12s:add(nn.JoinTable(3))				-- Add to the outgoing skip connection 'accumulator'

			local t12sj = nn.Sequential()		-- Container for handling the incoming skip connection
			t12sj:add(nn.NarrowTable(2, 2))	-- Select only the LSTM output and the forwarded network input.
			t12sj:add(nn.JoinTable(3))			-- Join network input and LSTM input 

			t12:add(t12s)										-- Handles outgoing skip connection to accumulator
			t12:add(t12sj)									-- For incoming skip connection to next LSTM layer.
			t12:add(nn.SelectTable(3))			-- Forward input for use in future incoming skip connections.
				-- Output from t12: table of 3 elements:
				-- {Output from outgoing skipcon accumulator, LSTM output + network input, network input}
			t1:add(t11)
			t1:add(t12)		-- Construct the complete layer.

			self.net:add(t1)	-- Add the completed layer to the overall network container.
		end
		--[[
		-- Batch normalisation
    if self.batchnorm == 1 then
      local view_in = nn.View(1, 1, -1):setNumInputDims(3)
      table.insert(self.bn_view_in, view_in)
      self.net:add(view_in)
      self.net:add(nn.BatchNormalization(H))
      local view_out = nn.View(1, -1):setNumInputDims(2)
      table.insert(self.bn_view_out, view_out)
      self.net:add(view_out)
    end
		
		-- Dropout
    if self.dropout > 0 then
      self.net:add(nn.Dropout(self.dropout))
    end
		--]]
  end
	
	self.net:add(nn.SelectTable(1))		-- This contains the outgoing skip connection accumulator (a large table)

  -- After all the RNNs run, we will have a tensor of shape (N, T, H);
  -- we want to apply a 1D temporal convolution to predict scores for each
  -- vocab element, giving a tensor of shape (N, T, V). Unfortunately
  -- nn.TemporalConvolution is SUPER slow, so instead we will use a pair of
  -- views (N, T, H) -> (NT, H) and (NT, V) -> (N, T, V) with a nn.Linear in
  -- between. Unfortunately N and T can change on every minibatch, so we need
  -- to set them in the forward pass.
	--
	-- MODIFICATION:
	-- Introducing skip connections means that after running, the tensor is of shape (N, T, self.num_layers * H)
	-- The same approach is used as below, but will be modified so that the layers are:
	-- view (N, T, self.num_layers * H) -> (NT, self.num_layers * H)
	-- nn.Linear(self.num_layers * H, V)
	-- view (NT, V) -> (N, T, V)
	--
  self.view1 = nn.View(1, 1, -1):setNumInputDims(3)
  self.view2 = nn.View(1, -1):setNumInputDims(2)

  self.net:add(self.view1)
  self.net:add(nn.Linear(self.num_layers * H, V))
  self.net:add(self.view2)

	print(self.net)
	--[[DEBUGGING--]]
	--print("BEGIN DEBUG CODE")

	--testIn = torch.ones(2, V)
	--self.net:forward(testIn)


	--print(#self.net.modules[4].output)
	--print("END DEBUG CODE")

end


function LM:updateOutput(input)
  local N, T = input:size(1), input:size(2)
  self.view1:resetSize(N * T, -1)
  self.view2:resetSize(N, T, -1)

  for _, view_in in ipairs(self.bn_view_in) do
    view_in:resetSize(N * T, -1)
  end
  for _, view_out in ipairs(self.bn_view_out) do
    view_out:resetSize(N, T, -1)
  end

  return self.net:forward(input)
end


function LM:backward(input, gradOutput, scale)
	print("gradOutput size: ", #gradOutput)
	return self.net:backward(input, gradOutput, scale)
end


function LM:parameters()
  return self.net:parameters()
end


function LM:resetStates()
  for i, rnn in ipairs(self.rnns) do
    rnn:resetStates()
  end
end


function LM:encode_string(s)
  local encoded = torch.LongTensor(#s)
  for i = 1, #s do
    local token = s:sub(i, i)
    local idx = self.token_to_idx[token]
    assert(idx ~= nil, 'Got invalid idx')
    encoded[i] = idx
  end
  return encoded
end


function LM:decode_string(encoded)
  assert(torch.isTensor(encoded) and encoded:dim() == 1)
  local s = ''
  for i = 1, encoded:size(1) do
    local idx = encoded[i]
    local token = self.idx_to_token[idx]
    s = s .. token
  end
  return s
end


--[[
Sample from the language model. Note that this will reset the states of the
underlying RNNs.

Inputs:
- init: String of length T0
- max_length: Number of characters to sample

Returns:
- sampled: (1, max_length) array of integers, where the first part is init.
--]]
function LM:sample(kwargs)
  local T = utils.get_kwarg(kwargs, 'length', 100)
  local start_text = utils.get_kwarg(kwargs, 'start_text', '')
  local verbose = utils.get_kwarg(kwargs, 'verbose', 0)
  local sample = utils.get_kwarg(kwargs, 'sample', 1)
  local temperature = utils.get_kwarg(kwargs, 'temperature', 1)

  local sampled = torch.LongTensor(1, T)
  self:resetStates()

  local scores, first_t
  if #start_text > 0 then
    if verbose > 0 then
      print('Seeding with: "' .. start_text .. '"')
    end
    local x = self:encode_string(start_text):view(1, -1)
    local T0 = x:size(2)
    sampled[{{}, {1, T0}}]:copy(x)
    scores = self:forward(x)[{{}, {T0, T0}}]
    first_t = T0 + 1
  else
    if verbose > 0 then
      print('Seeding with uniform probabilities')
    end
    local w = self.net:get(1).weight
    scores = w.new(1, 1, self.vocab_size):fill(1)
    first_t = 1
  end
  
  for t = first_t, T do
    if sample == 0 then
      local _, next_char = scores:max(3)
      next_char = next_char[{{}, {}, 1}]
    else
       local probs = torch.div(scores, temperature):double():exp():squeeze()
       probs:div(torch.sum(probs))
       next_char = torch.multinomial(probs, 1):view(1, 1)
    end
    sampled[{{}, {t, t}}]:copy(next_char)
    scores = self:forward(next_char)
  end

  self:resetStates()
  return self:decode_string(sampled[1])
end


function LM:clearState()
  self.net:clearState()
end
