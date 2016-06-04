require 'torch'
require 'nn'

require 'VanillaRNN'
require 'LSTM'
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
	self.batch_size = utils.get_kwarg(kwargs, 'batch_size')
	self.con_length = utils.get_kwarg(kwargs, 'con_length')	-- Argument to take the size of the constraint window, default 3100.

  local V, D, H = self.vocab_size, self.wordvec_dim, self.rnn_size
	local U = self.con_length	-- Setting the constraint vector size in the command line.
	local N = self.batch_size
	
	--[[Defining the temporary/storage variables--]]
	self.prev_w = {}	--torch.Tensor(N, T, D):fill(0)	-- At the start this is filled with 0...
	self.curr_w = {}	--torch.tensor(N, T, D):fill(0)
	self.dec_in = {}
	self.dec_con = {}
	self.LSTM1_out = {}
	self.a = {}
	self.b = {}
	self.k = {}
	self.exp_k_bar = {}
	self.net = nn.Sequential()	-- This is used for the jointable and views at the end.
	self.phiNetOut = {}
	self.LSTM2SeqOut = {}

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
	-- INPUT: single matrix containing x, and c concatenated together, dim (N + U, T, V)
	-- decodeNet: Split input apart, run through the lookup table.
	-- 		After lookupTable, need to stretch the constraint vectors so there are enough of them.
	-- 		(N+U, T, V) -> {(N, T, D), (N, U, T, D)}, for input and constraint vector respectively
	-- 
	--
	-- WINDOW CELL: 2 inputs, {input and C}
	-- 	2 sub-networks:
	--		net1 : PARAMETER HANDLING
	--				dimensions: {(N, T, H), (U, T, D}) -> {{(N, T, D), (N, T, D), (N, T, D)}, (U, T, D)}
	--				input: output from first LSTM hidden layer, C.
	--				outputs: {{alpha, beta, exp_k_bar}, C} (C is forwarded).
	--					NB: alpha and beta go in self.a and self.b
	--							self.k = self.k + exp_k_bar (needs to be done elementwise)
	--		
	--		net2 : PHI GENERATION
	--				dimensions: {{(N, T, D), (N, T, D), (N, T, D)}, (U, T, D)} -> {(N, T, U), (U, T, D)}
	--				input: table of (table of parameters alpha, beta, k.), and C
	--				output: table of Phi (for all T and all U, in one tensor), C (C is forwarded)
	--
	--		net3 : PHI APPLICATION (TO BE MERGED INTO net2)
	--				dimensions: {(N, T, U), (N, T, D, U)} -> (N, T, D)
	--				inputs: table of {Phi, C}
	--				outputs: window vector (to be fed into a hidden layer somewhere).
	

	self.decodeNet = nn.Sequential() 	-- Network containing the decoder, returns table of {x, c} decoded.

	--[[ Holders for quick reference of the hidden cells and the batch normalisation views --]]
	self.rnns = {}
  self.bn_view_in = {}
  self.bn_view_out = {}

 	-- V is the vocabulary size (number of symbols)
	-- D is the size of each individual one-hot vector
	-- N and U are minibatch and constraint vector size.
	--
	-- This lookup table converts a symbol (one of V possible symbols) to a vector of size D.
	-- Note that as long as the vocab size is the same, this just doesn't care about batch size.
	--	If input is size (N, T) then it should output (N, T, D)
	--	To extract C (which is appended to N) - slice over dimension 1.
  
	--[[ Construct a layer to decode both inputs in the input table. --]]
	self.decodeNet:add(nn.LookupTable(V, D))	-- Decodes all input, needs slicing along dimension 1.
		-- Output shape (N+U, T, D)
	local d1 = nn.ConcatTable()
	local d11 = nn.Narrow(1, 1, N)	-- Narrow along dimension 1 from index 1 to N
	local d12Con = nn.Sequential()
	local d12 = nn.Narrow(1, (N+1), (N+U))	-- Narrow along dimension 1 from index N+1 to N+U
	d12Con:add(d12)
	d12Con:add(nn.Unsqueeze(1))
	d12Con:add(nn.Replicate(N, 1))	-- Expand constraint into 4th dimension - (U, T, D) -> (N, U, T, D)
	d12Con:add(nn.Transpose(2, 3))
	d12Con:add(nn.Transpose(3, 4))	-- (N, U, T, D) -> (N, T, D, U) for later use.
	d1:add(d11)
	d1:add(d12Con)
	self.decodeNet:add(d1)
		-- Output at this point is {decoded input(N, T, D), decoded C (N, T, D, U)}
	
	--[[ Construct Parameter finding network --]]
  self.paramNet = nn.Sequential()		-- Network to calculate the parameters of the window.
	
  self.paramView1 = nn.View(1, 1, -1):setNumInputDims(3)
  self.paramView2 = nn.View(1, -1):setNumInputDims(2)			-- These are set every run in nn.updateOutputs.

  self.paramNet:add(self.paramView1)
  self.paramNet:add(nn.Linear(H, 3 * D))
  self.paramNet:add(self.paramView2)		-- (N, T, 3D)
	
	local pC = nn.ConcatTable()
	local pC1 = nn.Sequential()
	local pC2 = nn.Sequential()
	local pC3 = nn.Sequential()

	pC1:add(nn.Narrow(3, 1, D))
	pC1:add(nn.Exp())
	pC2:add(nn.Narrow(3, D+1, 2*D))
	pC2:add(nn.Exp())
	pC3:add(nn.Narrow(3, (2*D)+1, 3*D))
	pC3:add(nn.Exp())
	
	pC:add(pC1)	-- alpha : (N, T, D)
	pC:add(pC2)	-- beta : (N, T, D)
	pC:add(pC3)	-- exp_k_bar : (N, T, D)
			-- Note after processing, k will be (N, T, D)
	self.paramNet:add(pC)
	--paramNet outputs a TABLE
	--{a, b, exp_k_bar}

	--[[ Construct parameter -> phi network --]]
	--Input is a table of 3 things: {a, b, k}
	--These will have been set manually in the break.
	-- All should be of dimension (N, T, D)
	self.phiNet = nn.Sequential()
	local phiCol1 = nn.Sequential()		-- Input is self.a, dim (N, T, D)
	local phiCol2 = nn.Sequential()		-- Input is self.b, dim (N, T, D)
	local phiCol3 = nn.Sequential()		-- Input is self.k, dim (N, T, D)
	phiCol1:add(nn.SelectTable(1))
	phiCol2:add(nn.SelectTable(2))
	phiCol3:add(nn.SelectTable(3))
	local phiCol1Rep = nn.Replicate(U, 4)	
	local phiCol2Rep = nn.Replicate(U, 4)	
	local phiCol3Rep = nn.Replicate(U, 4)	
	phiCol1:add(phiCol1Rep)
	phiCol2:add(phiCol2Rep)
	phiCol3:add(phiCol3Rep)
	
	phiCol2:add(nn.MulConstant(-1, true)) -- beta has a factor of -1 in the phi equation.
	
	local phiCol3Concat = nn.Concat(4)
	for i=0,U do		-- This should implement the subtraction of u from k in the phi equation.
		local tSeq = nn.Sequential()
		tSeq:add(nn.Select(4, i))
		tSeq:add(nn.AddConstant(-i, true))
		tSeq:add(nn.Unsqueeze(4))
		phiCol3Concat:add(tSeq)
	end
	
	phiCol3:add(phiCol3Concat)
	phiCol3:add(nn.Square())

	local phiUpperConcatWrapper = nn.ConcatTable()
	phiUpperConcatWrapper:add(phiCol1)
	phiUpperConcatWrapper:add(phiCol2)
	phiUpperConcatWrapper:add(phiCol3)	-- phiConcatWrapper goes down to just above the nn.CMulTable()
	
	local phiLowerConcatWrapper = nn.ConcatTable()
	local phiLowCol1 = nn.Sequential()
	local phiLowCol2 = nn.Sequential()
	phiLowCol1:add(nn.SelectTable(1))
	phiLowCol1:add(nn.Identity())
	phiLowCol2:add(nn.NarrowTable(2, 2))
	phiLowCol2:add(nn.CMulTable())
	phiLowCol2:add(nn.Exp())
	
	phiLowerConcatWrapper:add(phiLowCol1)
	phiLowerConcatWrapper:add(phiLowCol2)

	self.phiNet:add(phiUpperConcatWrapper)
	self.phiNet:add(phiLowerConcatWrapper)	-- outputs a table of 2 elements
	self.phiNet:add(nn.CMulTable())
	self.phiNet:add(nn.Sum(3, 4))	-- Sum over D.

	self.phiNet:add(nn.Unsqueeze(3, 4))
	self.phiNet:add(nn.Replicate(D, 3))

	--[[Build last layer out of phi, needs a C input for the actual mixture. --]]
	self.phiOut = nn.Sequential()	-- Should be fed a table: {C, phi_altered}, dimensions {(N, T, D, U), (N, T, D, U)}
	self.phiOut:add(nn.CMulTable())
	self.phiOut:add(nn.Sum(4))
		-- Output should be of shape (N, T, D))
	
	--[[ Build any possibly high level structure --]]
	--[[ Built modules:
	--		self.decodeNet (in: input, out: {encoded input, encoded constraint})
	--					(N+U, T, V) -> {(N, T, D), (N, T, D, U)}, for input and constraint vector respectively
	--		self.paramNet (in: LSTM1 output, out: {alpha, beta, exp_k_bar})
	--					(N, T, H) -> {(N, T, D), (N, T, D), (N, T, D)}
	--		self.phiNet (in: {alpha, beta, k}, out: phi.)
	--					{(N, T, D), (N, T, D), (N, T, D)} -> (N, T, U)
	--		self.phiOut (in: {C, Phi}, out: w)
	--					{(N, T, D, U), (N, T, U)} -> (N, T, D)
	--]]

	local LSTM1 = nn.LSTM((2*D), H)	-- First hidden layer
	local LSTM2 = nn.LSTM(H+D+D, H)	-- Second hidden layer, batchnorm and dropout disabled for brevity.
	LSTM1.remember_states = true
	LSTM2.remember_states = true
	table.insert(self.rnns, LSTM1)
	table.insert(self.rnns, LSTM2)
	
	-- Make 2 modules to hold the LSTMs and their input concatenators
	self.LSTM1Seq = nn.Sequential()
	self.LSTM2Seq = nn.Sequential()
	self.LSTM1Seq:add(nn.JoinTable(3))
	self.LSTM1Seq:add(LSTM1)
	self.LSTM2Seq:add(nn.JoinTable(3))
	self.LSTM2Seq:add(LSTM2)
	
	--[[ CONSTRUCT self.net1 --]]
	--[[
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
	--]]
  -- After all the RNNs run, we will have a tensor of shape (N, T, H);
  -- we want to apply a 1D temporal convolution to predict scores for each
  -- vocab element, giving a tensor of shape (N, T, V). Unfortunately
  -- nn.TemporalConvolution is SUPER slow, so instead we will use a pair of
  -- views (N, T, H) -> (NT, H) and (NT, V) -> (N, T, V) with a nn.Linear in
  -- between. Unfortunately N and T can change on every minibatch, so we need
  -- to set them in the forward pass.
	
  self.view1 = nn.View(1, 1, -1):setNumInputDims(3)
  self.view2 = nn.View(1, -1):setNumInputDims(2)

  self.net:add(self.view1)
  self.net:add(nn.Linear(H, V))
  self.net:add(self.view2)

end

function LM:updateOutput(input)
  local N, T = self.batch_size, input:size(2)
	local D = self.wordvec_dim
	local U = input:size(1) - N
  -- Set the view sizes according to the batch size and number of timesteps to unroll
	self.view1:resetSize(N * T, -1)
  self.view2:resetSize(N, T, -1)
	
	-- Set paramView up the same way
	self.paramView1:resetSize(N * T, -1)
	self.paramView2:resetSize(N, T, -1)
	
	if #self.curr_w == 0 then
		self.curr_w = torch.Tensor(N, T, D):fill(0)
		self.prev_w = torch.Tensor(N, T, D):fill(0)
	end

	--[[
	-- Set up the batch normalisation views according to the current N and T.
  for _, view_in in ipairs(self.bn_view_in) do
    view_in:resetSize(N * T, -1)
  end
  for _, view_out in ipairs(self.bn_view_out) do
    view_out:resetSize(N, T, -1)
  end
	--]]
	--[[ Built modules:
	--		self.decodeNet (in: input, out: {encoded input, encoded constraint})
	--					(N+U, T, V) -> {(N, T, D), (N, T, D, U)}, for input and constraint vector respectively
	--		self.paramNet (in: LSTM1 output, out: {alpha, beta, exp_k_bar})
	--					(N, T, H) -> {(N, T, D), (N, T, D), (N, T, D)}
	--		self.phiNet (in: {alpha, beta, k}, out: phi.)
	--					{(N, T, D), (N, T, D), (N, T, D)} -> (N, T, U)
	--		self.phiOut (in: {C, Phi}, out: w)
	--					{(N, T, D, U), (N, T, U)} -> (N, T, D)
	--		self.LSTM1Seq = nn.Sequential()
	--					{prev_w, dec_in} -> (N, T, 2D)
	--		self.LSTM2Seq = nn.Sequential()
	--]]

	-- START: input
	local decodeFOut = self.decodeNet:forward(input)
	self.dec_in, self.dec_con = decodeFOut[1], decodeFOut[0]

	local LSTM1NetOut = self.LSTM1Seq:forward({self.curr_w, self.dec_in})
	self.LSTM1_out = {LSTM1NetOut[1], LSTM1NetOut[2]}

	local paramNetOut = self.paramNet:forward(self.LSTM1_out)
	self.a, self.b, self.exp_k_bar = paramNetOut[1], paramNetOut[2], paramNetOut[3]
	self.k = self.k + self.exp_k_bar

	self.phiNetOut = self.phiNet:forward({self.a, self.b, self.k})
	self.prev_w = self.curr_w
	self.curr_w =  self.phiOut:forward({self.phiNetOut, self.dec_con})
	self.LSTM2SeqOut = self.LSTM2Seq:forward({self.LSTM1_out, self.curr_w, self.dec_in})
	return self.net:forward(self.LSTM2SeqOut)
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
	--
	--When two backwards modules join (e.g. between LSTM1 and paramNet), get both gradInputs and put them in a table
	--i.e. net:backward(in, {gradInput1, gradInput2}
	
	self.net:backward(self.LSTM2SeqOut, gradOutput, scale)
	self.LSTM2Seq:backward({self.LSTM1_out, self.curr_w, self.dec_in}, self.net.gradInput, scale)
	
	self.phiOut:backward({self.phiNetOut, self.dec_con}, self.LSTM2Seq.gradInput, scale)
	self.phiNet:backward({self.a, self.b, self.k}, self.phiOut.gradInput, scale)
	self.paramNet:backward(self.LSTM1_out, self.phiNet.gradInput, scale)
	self.LSTM1Seq:backward({self.prev_w, self.dec_in}, {self.paramNet.gradInput, self.LSTM2Seq.gradInput}, scale)
	return self.decodeNet:backward(input, {self.LSTM1Seq.gradInput, self.LSTM2Seq.gradInput, self.phiOut.gradInput}, scale)
	
	--return self.net:backward(input, gradOutput, scale)
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
