require 'torch'
require 'nn'

require 'VanillaRNN'
require 'LSTM'
require 'windowedLSTM'
require 'dimPrint'		-- Self-written debugging module whose purpose is to print the dimensions of tensors passing through.

local utils = require 'util.utils'


local LM, parent = torch.class('nn.LanguageModelSkip_win1', 'nn.Module')


function LM:__init(kwargs)
  self.idx_to_token = utils.get_kwarg(kwargs, 'idx_to_token')
  self.token_to_idx = {}
  self.vocab_size = 0
  for idx, token in pairs(self.idx_to_token) do
    self.token_to_idx[token] = idx
    self.vocab_size = self.vocab_size + 1
  end

  print("Using a fixed architecture - 2 hidden LSTM layers + windowing.")
  -- Model type and number of layers are forced to lstm and 2 respectively.
	self.model_type = 'lstm'
  self.wordvec_dim = utils.get_kwarg(kwargs, 'wordvec_size')
  self.rnn_size = utils.get_kwarg(kwargs, 'rnn_size')
	self.num_layers = 2
  self.dropout = utils.get_kwarg(kwargs, 'dropout')
  self.batchnorm = utils.get_kwarg(kwargs, 'batchnorm')

  local V, D, H = self.vocab_size, self.wordvec_dim, self.rnn_size

	--[[ Adding the Window of Graves Et. Al --]]
	-- The window:
	-- 	window at time t (w_t) is sum over u of phi(t, u)*c_u
	-- 	i.e. The sum over all character (one hot vectors) multiplied by phi(t, u)
	-- 		c_u		: character u of constraint vector c
	-- 		t			: timestep
	-- 	phi(t, u):
	-- 		Sum of K of these:
	-- 				alpha_k_t * exp(-beta_k_t * (k_k_t - u)^2)
	--		K is the number of possible text classes
	--			(i.e. number of items in a single characters's one hot vector)
	--		t is the timestep (up to T)
	--	The parameters:
	--		Each parameter is a vector of size K
	--		=> all three make a vector of size 3K
	--		(alpha_t$, beta_t$, k_t$) = Linear(h_1 -> 3K)
	--			alpha_t = exp(alpha_t$)
	--			beta_t = exp(beta_t$)
	--			k_t = k_(t-1) + exp(k_t$)
	--
	--]]
	--
	--[[ IMPLEMENTATION NOTES --]]
	-- INPUT: single vector containing x, and c concatenated together.
	-- decodeNet: Split input apart, run through the lookup table.
	--
	-- WINDOW CELL:
	-- 	2 sub-networks:
	--		net1 : PARAMETER HANDLING
	--				dimensions: {(N, T, H), (N, T, D, U}) -> {{(N, T, D), (N, T, D), (N, T, D)}, (N, T, D, U)}
	--				input: output from first LSTM hidden layer, C.
	--				outputs: {{alpha, beta, exp_k_bar}, C} (C is forwarded).
	--					NB: alpha and beta go in self.a and self.b
	--							self.k = self.k + exp_k_bar (needs to be done elementwise)
	--		
	--		net2 : PHI GENERATION
	--				dimensions: {{(N, T, D), (N, T, D), (N, T, D)}, (N, T, D, U)} -> {(N, T, U), (N, T, D, U)}
	--				input: table of (table of parameters alpha, beta, k.), and C
	--				output: table of Phi (for all T and all U, in one tensor), C (C is forwarded)
	--
	--		net3 : PHI APPLICATION
	--				dimensions: {(N, T, U), (N, T, D, U)} -> (N, T, D)
	--				inputs: table of {Phi, C}
	--				outputs: window vector (to be fed into a hidden layer somewhere).
	

	self.decodeNet = nn.Sequential() 	-- Network containing the decoder, returns table of {x, c} decoded.
  self.net1 = nn.Sequential()
  self.net2 = nn.Sequential()
	self.rnns = {}
  self.bn_view_in = {}
  self.bn_view_out = {}

  local LUT = nn.LookupTable(V, D)
	--self.net:add(nn.LookupTable(V, D))
 	-- V is the vocabulary size (number of symbols)
	-- D is the size of each individual one-hot vector
	-- This lookup table converts a symbol (one of V possible symbols) to a vector of size D.
	-- Note that as long as the vocab size is the same, this just doesn't care about batch size.

	-- Construct a layer to decode both inputs in the input table.
	-- Input to the whole network will be a CONCATENATED TENSOR: input x (dim V) + constraint c (dim V_C)
	local t1 = nn.ConcatTable()
	local t11 = nn.Sequential()
	local t12 = nn.Sequential()
	
	--[[ CONSTRUCT self.decodeNet --]]
	self.InSplitC = nn.Narrow(3, V+1, V+2)	-- THIS WILL BE RESET EVERY ITERATION in updateOutputs
	--t11:add(nn.SelectTable(1))	-- Selecting the input x
	t11:add(nn.Narrow(3, 1, V))		-- Select x from the input - dimensions 1 to V
	t11:add(LUT)
	--t12:add(nn.SelectTable(2))	-- Selecting the constraint vector c
	t12:add(self.InSplitC)
	t12:add(LUT)	-- Both inputs go through a copy of the same nn.LookupTable (I think)
			
	t1:add(t11)		-- Add the sublayers to the concatTable
	t1:add(t12)

	self.decodeNet:add(t1)	-- Add the concatTable to the network.


	--[[ CONSTRUCT self.net1 --]]
	for i = 1, self.num_layers do
    -- Selecting input dimensions for LSTM cells
		local prev_dim = D+H								
    local out_dim = H
		if i == 1 then prev_dim = D end			-- First layer LSTM has input dimension D only
   	if i == 1 then out_dim = D+H end
		-- Selecting cell type to use
		local rnn
    local rnnContainer = nn.Sequential()
		
		-- Choose which LSTM type to use - windowed or unwindowed - depending on the layer number
		-- The windowed layer gets put into self.net1
		if i == 1 then
			rnn = nn.windowedLSTM(prev_dim, out_dim)			-- Constructor takes input and output sizes.
				-- nn.windowedLSTM has a control vector of size D concatenated to the normal LSTM output H.
		elseif i~= 1 then
			rnn = nn.LSTM(prev_dim, out_dim)			-- Constructor takes input and output sizes.
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
		-- Note that after encoding, the table is {encoded data, encoded constraint}	
		if i == 1 then -- Set up the first layer
			local l1 = nn.ConcatTable()
			l1:add(rnnContainer)				-- Add windowed LSTM, which takes a table {encoded x, encoded c} as input.
			l1:add(nn.SelectTable(1))		-- Add selector to ensure that input gets forwarded for skips.
			self.net:add(l1)

		elseif i ~= 1 then --Set up the second layer - this network should only really have 2.
			local l2 = nn.Sequential()
			l2:add(nn.JoinTable(3, 3))
			l2:add(rnnContainer)
			self.net:add(l2)
		end
  end
	
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
  self.net:add(nn.Linear(H, V))
  self.net:add(self.view2)

end

function LM:updateOutput(input)
  -- Set the view sizes according to the batch size and number of timesteps to unroll
	local N, T = input:size(1), input:size(2)
  self.view1:resetSize(N * T, -1)
  self.view2:resetSize(N, T, -1)

	-- Set the input splitter up according to the size of the input.
	local LC = input:size()
	print(LC)
	print(self.vocab_size)
	self.InSplitC = nn.Narrow(3, self.vocab_size + 1, LC)	

	-- Set up the batch normalisation views according to the current N and T.
  for _, view_in in ipairs(self.bn_view_in) do
    view_in:resetSize(N * T, -1)
  end
  for _, view_out in ipairs(self.bn_view_out) do
    view_out:resetSize(N, T, -1)
  end

  return self.net:forward(input)
end

function LM:backward(input, gradOutput, scale)
	--TODO: Modify to link the sub-networks together
	--Note that :
	--	1) net:forwards(input) must have already been called
	--	2) net:backward(input, gradOutput, scale) must be called WITH THE SAME INPUT AS FORWARDS.
	--	3) The 'gradOutput' input must be equal to the (internal) state 'gradInput' from the next network
	--			in order (i.e. the most recently backward's'ded. That's a word. Sure.
	--EACH BACKWARDS MODIFIES NET INTERNAL gradInput
	--gradInput must be passed as the gradOutput of the network that is next to be backpropagated through
	--e.g. if whole network is net 1 then net 2:
	--net2:backward(input, gradOutput, scale)
	--	-- Assume that the intermediary value (forward output of net1) is self.n12Temp
	--net1:backward(n12Temp, net2.gradInput, scale)
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
