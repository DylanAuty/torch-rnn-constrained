require 'torch'
require 'nn'
require 'nngraph'

require 'VanillaRNN'
require 'LSTM'

local utils = require 'util.utils'

nngraph.setDebug(true)


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
  --self.net = nn.Sequential()
  self.rnns = {}					-- This is used by LM:resetStates to clear all LSTM/RNN states.
  self.bn_view_in = {}	
  self.bn_view_out = {}
	
	--[[ Defining modules to be used in the graph --]]
	
  --self.net:add(nn.LookupTable(V, D))	-- Originally this was the first layer of the nn.Sequential container.
  local LUT = nn.LookupTable(V, D)()		-- This is the first thing in the graph.
	-- Creating a skip connected LSTM with self.num_layers hidden layers
	
	local rnn
	local jt

	for i = 1, self.num_layers do
		local prev_dim = D+H							-- All layers above the first layer have input size D+H
		
		-- Select the model type, VanillaRNN.lua or LSTM.lua
		-- All layers will have output size H.
		if i == 1 then	-- On first layer, connect first layer to input (rather than previous layer)
			prev_dim = D	-- First layer input size is only D
			if self.model_type == 'rnn' then
				rnn = nn.VanillaRNN(prev_dim, H)(LUT)
			elseif self.model_type == 'lstm' then
				rnn = nn.LSTM(prev_dim, H)(LUT)
			end
			-- Set the layer up and insert it into a reference table of all layers
			rnn.remember_states = true
			table.insert(self.rnns, rnn)
			
			jt = nn.Identity()(rnn)

			-- self.net:add(rnn)	-- This line was used when self.net was an nn.Sequential().
			--[[BATCHNORM GOES HERE --]]

			--[[DROPOUT GOES HERE --]]	
		
		elseif i ~= 1 then
			--DEBUGLINE
			prev_dim = H
			--/DEBUGLINE
			if self.model_type == 'rnn' then
				rnn = nn.VanillaRNN(prev_dim, H)(nn.JoinTable(2)({rnn, LUT})[{{}, {}, 1}])
			elseif self.model_type == 'lstm' then
				--rnn = nn.LSTM(prev_dim, H)(nn.JoinTable(3)({rnn, LUT}))
				rnn = nn.LSTM(prev_dim, H)(rnn)
			end
			-- Set the layer up and insert it into a reference table of all layers
			rnn.remember_states = true
			table.insert(self.rnns, rnn)

			jt = nn.JoinTable(3)({jt, rnn})
			-- self.net:add(rnn)	-- This line was used when self.net was an nn.Sequential().
			--[[BATCHNORM GOES HERE --]]

			--[[DROPOUT GOES HERE --]]	
		end
	end
  
	-- At the end of this, jt contains the outputs and all the skip connections - we need to reign in the dimensions.
	-- Tensor will be shape (N, T, num_layers*H) due to skip connections
	-- To deal with nn.Linear, we need to convert dimensions using views as below:
	-- (N,T,num_layer*H) -> (NT, num_layers*H) -> {nn.Linear(num_layers * H, H)} -> (N, T, H)
	
	compactorContainer = nn.Sequential()
	--[[
	Cview1 = nn.View(1, 1, -1):setNumInputDims(3)
	Cview2 = nn.View(1, -1):setNumInputDims(2)
	compactorContainer:add(Cview1)
	compactorContainer:add(nn.Linear(self.num_layers * H, H))
	compactorContainer:add(nn.SoftMax())	-- Graves Et Al, 2013 specifies an output function - but doesn't say what it is. Softmax used instead.
	compactorContainer:add(Cview2)
	]]--

	-- After all the RNNs run, we will have a tensor of shape (N, T, H);
  -- we want to apply a 1D temporal convolution to predict scores for each
  -- vocab element, giving a tensor of shape (N, T, V). Unfortunately
  -- nn.TemporalConvolution is SUPER slow, so instead we will use a pair of
  -- views (N, T, H) -> (NT, H) and (NT, V) -> (N, T, V) with a nn.Linear in
  -- between. Unfortunately N and T can change on every minibatch, so we need
  -- to set them in the forward pass.
	convoContainer = nn.Sequential()
	
	self.view1 = nn.View(1, 1, -1):setNumInputDims(3)
	self.view2 = nn.View(1, -1):setNumInputDims(2)
	convoContainer:add(self.view1)
	convoContainer:add(nn.Linear(self.num_layers * H, V))
	convoContainer:add(self.view2)
	-- Plug together: rnn -> view1 -> Linear(H, V) -> view2
	--y = nn.Identity()(convoContainer(compactorContainer(jt)))
	y = nn.Identity()(convoContainer(jt))
	self.net = nn.gModule({LUT}, {y})

	graph.dot(self.net.fg, 'Built1', 'debug1fg')
	--[[
	-- Copy of original loop below
	--
	for i = 1, self.num_layers do
    local prev_dim = H
    if i == 1 then prev_dim = D end
    local rnn
    if self.model_type == 'rnn' then
      rnn = nn.VanillaRNN(prev_dim, H)
    elseif self.model_type == 'lstm' then
      rnn = nn.LSTM(prev_dim, H)
    end
    rnn.remember_states = true
    table.insert(self.rnns, rnn)
    self.net:add(rnn)
    if self.batchnorm == 1 then
      local view_in = nn.View(1, 1, -1):setNumInputDims(3)
      table.insert(self.bn_view_in, view_in)
      self.net:add(view_in)
      self.net:add(nn.BatchNormalization(H))
      local view_out = nn.View(1, -1):setNumInputDims(2)
      table.insert(self.bn_view_out, view_out)
      self.net:add(view_out)
    end
    if self.dropout > 0 then
      self.net:add(nn.Dropout(self.dropout))
    end
  end
	--]]
	
	--[[
  self.view1 = nn.View(1, 1, -1):setNumInputDims(3)
  self.view2 = nn.View(1, -1):setNumInputDims(2)
	
  self.net:add(self.view1)
  self.net:add(nn.Linear(H, V))
  self.net:add(self.view2)
	]]--
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
  print("Running backwards")
	return self.net:backward(input, gradOutput, scale)
end


function LM:parameters()
  return self.net:parameters()
end


function LM:resetStates()
  -- Iterates over every RNN or LSTM in the network and resets the states.
	for i, rnn in ipairs(self.rnns) do
    rnn:resetStates()	-- This is a method contained within the LSTM/RNN definition
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
