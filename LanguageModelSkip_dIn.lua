require 'torch'
require 'nn'

require 'VanillaRNN'
require 'LSTM'
require 'dimPrint'		-- Self-written debugging module whose purpose is to print the dimensions of tensors passing through.

local utils = require 'util.utils'


local LM, parent = torch.class('nn.LanguageModelSkip_dIn', 'nn.Module')


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

  -- This model is identical to the skip connected network (LanguageModelSkipCon.lua), except
	-- It expects to receive a 47 long vector as the input, for every corresponding output character
	-- In the normal network, input is (N, T) tensor of character indices - they get put through a LUT
	-- Here the data has already been decoded
	-- An input batch is therefore of size (N, T, 47) instead of (N, T) as it is when taking character input
	-- As a result, removing the nn.LookupTable should sort it out.
	-- Also dimensions of the hidden cells need to be changed accordingly.

	local V, H = self.vocab_size, self.rnn_size
	local D = 47	-- Hard setting the input vector, data is a 47 wide vector
	
  self.net = nn.Sequential()
  self.rnns = {}
  self.bn_view_in = {}
  self.bn_view_out = {}

  --self.net:add(nn.LookupTable(V, D))	
 	
	for i = 1, self.num_layers do
    -- Selecting input dimensions for LSTM cells
		local prev_dim = D+H								-- All LSTMs in layers 2 onwards have input dimension D+H (H because of skip connections)
    if i == 1 then prev_dim = D end			-- First layer LSTM has input dimension D only
   	
		-- Selecting cell type to use
		local rnn
    local rnnContainer = nn.Sequential()
		if self.model_type == 'rnn' then
      rnn = nn.VanillaRNN(prev_dim, H)
    elseif self.model_type == 'lstm' then
      rnn = nn.LSTM(prev_dim, H)
    end
    rnn.remember_states = true
    table.insert(self.rnns, rnn)				-- This table is used to find all the rnn cells and reset them later.
 		rnnContainer:add(rnn)
		
		-- Batch normalisation
    if self.batchnorm == 1 then
      local view_in = nn.View(1, 1, -1):setNumInputDims(3)
      table.insert(self.bn_view_in, view_in)
      rnnContainer:add(view_in)
      rnnContainer:add(nn.BatchNormalization(H))
      local view_out = nn.View(1, -1):setNumInputDims(2)
      table.insert(self.bn_view_out, view_out)
      rnnContainer:add(view_out)
    end
		
		-- Dropout
    if self.dropout > 0 then
      rnnContainer:add(nn.Dropout(self.dropout))
    end


		-- Construct and link the layers
		
		if i == 1 then	-- Set up first layer
			local t1 = nn.Sequential()			-- Contains both sub-layers
			local t11 = nn.ConcatTable()		-- First sub-layer
			local t12 = nn.ConcatTable()		-- Second sub-layer

			t11:add(rnnContainer)
			t11:add(nn.Identity())		-- output 1 of this layer is LSTM output, output 2 is input.
				-- Output from t11: table of 2 elements:
				-- 	{LSTM output, network input}
			t12:add(nn.SelectTable(1))		-- Grab LSTM output only from first sublayer.
			t12:add(nn.JoinTable(3, 3))			-- For incoming skip connection to next LSTM layer.
			t12:add(nn.SelectTable(2))		-- Forward input for use in future incoming skip connections.
				-- Output from t12: table of 3 elements:
				-- 	{LSTM output, LSTM output + network input, network input}
			t1:add(t11)
			t1:add(t12)		-- Construct the complete first layer.
			self.net:add(t1)	-- Add the completed layer to the overall network container.

		elseif i ~= 1 then	-- Set up any layers after the first layer
			local t1 = nn.Sequential()			-- Container for both sub-layers
			local t11 = nn.ConcatTable()		-- First sub-layer
			local t12 = nn.ConcatTable()		-- Second sub-layer
			
			-- Define sequentials to contain a SelectTable and a module.
			local seq1 = nn.Sequential()
			local seq2 = nn.Sequential()
			local seq3 = nn.Sequential()
			
			seq1:add(nn.SelectTable(1))
			seq2:add(nn.SelectTable(2))
			seq3:add(nn.SelectTable(3))	-- This bit replicates the nn.ParallelTable functionality
			
			seq1:add(nn.Identity())
			seq2:add(rnnContainer)			-- LSTM/RNN and Batchnorm and dropout if enabled
			seq3:add(nn.Identity())

			t11:add(seq1)
			t11:add(seq2)
			t11:add(seq3)								
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
	-- The same approach is used as in the original language model, but will be modified so that the layers are:
	-- view (N, T, self.num_layers * H) -> (NT, self.num_layers * H)
	-- nn.Linear(self.num_layers * H, V)
	-- view (NT, V) -> (N, T, V)
	--
  self.view1 = nn.View(1, 1, -1):setNumInputDims(3)
  self.view2 = nn.View(1, -1):setNumInputDims(2)

  self.net:add(self.view1)
  self.net:add(nn.Linear(self.num_layers * H, V))
  self.net:add(self.view2)

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
	local nullStop = utils.get_kwarg(kwargs, 'nullstop', 0)	-- Argument to stop sampling/truncate output after a null character is generated.
		
	if nullStop > 0 then	-- Change the sample limit if the nullStop argument is set to 1.
		T = 20000	-- Hardcoding this to be truncated later, should be adequate... HEY I'M SURE IT'S PROBABLY FINE RIGHT
	end
	
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
  
	local n1Flag = 0	-- Having to use flags to detect the string "\n\n." which should mean the end of a forecast.
	local n2Flag = 0

	-- Trying to remove a little overhead by repeating the loop twice - once for nullStop, once for no nullStop.
	if nullStop > 0 then
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
			
			if (n1Flag == 0 and n2Flag == 0) then	-- No newlines detected yet.
				if (self:decode_string(next_char[1]) == "\n") then
					n1Flag = 1
				end
			elseif (n1Flag == 1 and n2Flag == 0) then	-- First newline detected already
  			if (self:decode_string(next_char[1]) == "\n") then
					n2Flag = 1	-- Say that we've detected the second newline
				else
					n1Flag = 0	-- Reset since the latest character is just a normal newline.
				end
			elseif (n1Flag == 1 and n2Flag == 1) then -- 2 newlines detected in a row.
				if (self:decode_string(next_char[1]) == ".") then	-- This is the last of the "\n\n."
					sampled:resize(1, t-1)	-- Resize output vector to the final size
					break	-- If a null character is received then stop sampling. Don't write the character to the output here.
				else
					n1Flag = 0
					n2Flag = 0
				end
			end
		end
	else	-- Same thing, without the comparisons and truncation.
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
	end
	
  self:resetStates()
  return self:decode_string(sampled[1])
end


function LM:clearState()
  self.net:clearState()
end
