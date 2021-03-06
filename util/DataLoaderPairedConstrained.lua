require 'torch'
require 'hdf5'

local utils = require 'util.utils'

local DataLoader = torch.class('DataLoaderPairedConstrained')


function DataLoader:__init(kwargs)
  local h5_file = utils.get_kwarg(kwargs, 'input_h5')
  self.batch_size = utils.get_kwarg(kwargs, 'batch_size')
  self.seq_length = utils.get_kwarg(kwargs, 'seq_length')
  self.con_length = utils.get_kwarg(kwargs, 'con_length')
	local N, T = self.batch_size, self.seq_length
	local U = self.con_length
	-- Structure of the itemized h5:
	-- /
	-- /train
	-- /test
	-- /val
	-- /train/data
	-- /train/forecast		Are the groups
	-- /train/forecast/i 	Is the actual data

  -- Just slurp all the data into memory
  local splits = {}
	local trainData = {}	-- These 6 I want to fill with key/dataset pairs - Assign with tableName["keyhere"] = value
	local testData = {}		-- Ideally they should come out as a table of tensors....
	local valData = {}
	local trainForecasts = {}	
	local testForecasts = {}	
	local valForecasts  = {}
  local f = hdf5.open(h5_file, 'r')
  --splits.train = f:read('/train'):all()
  --splits.val = f:read('/val'):all()
  --splits.test = f:read('/test'):all()
	
	--print(f:read("/train/data"):all())	-- This returns a boatload of key/tensor pairs where the tensors are the data
	-- Write every training example to the trainEx table
	
	--for key, item in pairs(f:read('/train/data'):all()) do
	--	trainEx[key] = item
	--	for k2, item2 in pairs(trainEx) do
	--		print(#item2)
	--	end
	--	os.exit()
	--end	
	
	print("Loading data from HDF5 to Memory...")
	--[[
	trainData = f:read('/train/data'):all()
	trainForecasts = f:read('/train/forecast'):all()
	testData = f:read('/test/data'):all()
	trainForecasts = f:read('/test/forecast'):all()
	--]]
	valData = f:read('/val/data'):all()
	valForecasts = f:read('/val/forecast'):all()

	-- originally, splits contained 3 datasets - train, test, val
	-- Now it contains 3 groups of 2 groups each of many datasets.
	
	-- The splits
	--[[
	splits.train = {}
	splits.train.data = {}
	splits.train.forecasts = {}
	splits.test = {}
	splits.test.data = {}
	splits.test.forecasts = {}
	--]]
	splits.val = {}
	splits.val.data = {}
	splits.val.forecasts = {}
	
	print("Sorting the data into sets...")
	-- Fill the sets
	--[[
	for key, item in pairs(trainData) do
		splits.train.data[key] = item
		splits.train.forecasts[key] = trainForecasts[key]
	end
	for key, item in pairs(testData) do
		splits.test.data[key] = item
		splits.test.forecasts[key] = testForecasts[key]
	end
	--]]
	for key, item in pairs(valData) do
		splits.val.data[key] = item
		splits.val.forecasts[key] = valForecasts[key]
	end
	
	-- The sub-splits, one each for data and forecasts
	--splits.train.data = trainData
	--splits.train.forecasts = trainForecasts
	--splits.test.data = testData
	--splits.test.forecasts = testForecasts
	--splits.val.data = valData
	--splits.val.forecasts = valForecasts
	
  self.x_splits = {}
  self.y_splits = {}
  self.split_sizes = {}

	-- This loop constructs minibatches ready for use
  -- To make life simpler, aim is to keep this loop with the same output:
	-- 	Just x_splits, y_splits, split_sizes.
	-- All constraint stuff should be sorted out in here.
	for split, set in pairs(splits) do -- For every top level dataset split (train/test/val)
    -- Split is one of 3 sets of datasets
		local firstFlag = 0
		print("Reshaping data")
		for datasetNum, v in pairs(set.forecasts) do
			-- v is an array of encoded characters.
			local num = v:nElement()
			local extra = num % (N * T)
			-- num is number of chars in the split
			-- N is batch length
			-- T is sequence length

			if extra == 0 then	-- If the batch size fits exactly, then we append one so that we can construct y properly.
				t1 = torch.ByteTensor(1)
				t1[1] = 0
				v = torch.cat(v, t1)
			end

    	-- Chop out the extra bits at the end to make it evenly divide
			-- If it fits perfectly... append 
			-- vx is of dimension (E, N, T), E is the index of the training example
			-- 	Need to make it (E, N + C, T) where C is constraint size
			local vx = v[{{1, num - extra}}]:view(N, -1, T):transpose(1, 2):clone()	
			local vy = v[{{2, num - extra + 1}}]:view(N, -1, T):transpose(1, 2):clone()

			-- Now vx has dim (E, N, T)

			-- Now extract and append the constraint vector in the right place...
			-- Needs to be appended to N in every place
			-- Appending along the minibatch direction because it will be sliced off on arrival
			-- 	Every minibatch must come from the same example
			--	Every constraint appended to that minibatch should correspond to the same example.
			
			vc = set.data[datasetNum] -- vc is a vector of size (C)
			
			temp = torch.ByteTensor(U, vx:size(1), T):fill(0) -- Currently (C, E, T), will change later to (E, C, T)
			if (vc:nElement() > U) then	-- If there's more data than space in C, trim the data.
				-- Append vc to temp column by column
				-- This currently takes five billion years
				-- Would make more sense to repeat vc in the relevant dimensions, then append in one go.
				for x=1,vx:size(1) do	-- vx:size(1) = E.
					for y=1,T do	
						temp[{{}, x, y}] = vc[{{1, U}}]	-- truncate vc if too long
					end
				end
			else	-- If there's just enough or too much space for the data, only fill a bit of the data.
				for x=1,vx:size(1) do	-- vx:size(1) = E.
					for y=1,T do
						local temp2 = temp[{{1, U}, x, y}]
						temp2 = vc
					end
				end
			end
			temp = temp:transpose(1, 2)		-- (C, E, T) -> (E, C, T)
			-- After this loop, temp has dimensions (C, E, T)
			-- Append to vx (dim (E, N, T)) along 2nd dimension.
			-- We want an output of size (E, N + C, T)
			-- 				That way, out[1] has dim (N+C, T)
			-- 					=> for time t, it's a vector of (N+C)
			-- 		Think of V as indexing whole examples
			-- 		Each example consists of N + C characters.

			vx = torch.cat(vx, temp, 2) 			-- Now, the constraint is on the end of every minibatch. It can be chopped off on receipt.
			-- x and y are the input and reference output respectively
			-- y is the same as x, just transposed by 1
			-- Note that the outputs are indexed by split alone.
			-- x_splits and y_splits are empty tables, NOT TENSORS.
			if firstFlag == 0 then
				self.x_splits[split] = vx
				self.y_splits[split] = vy
				self.split_sizes[split] = vx:size(1)
			else
				self.x_splits[split] = torch.cat(vx, self.x_splits[split], 1)
    		self.y_splits[split] = torch.cat(vy, self.y_splits[split], 1)	-- Add to the vector of example slices
    		self.split_sizes[split] = self.split_sizes[split] + vx:size(1)
  		end
		end
	end

  self.split_idxs = {train=1, val=1, test=1}
end


function DataLoader:nextBatch(split)
	-- y will be the same as the input but shifted by one
  local idx = self.split_idxs[split]
	assert(idx, 'invalid split ' .. split)
  
	local x = self.x_splits[split][idx]	 		-- x of size (N+C, T)
	local y = self.y_splits[split][idx]			-- y of size (N, T)
			-- Intention is that receiving network unpacks x to retrieve C,
			-- but will know 100% that the C it gets is supposed to be with the minibatch of size N.
  if idx == self.split_sizes[split] then
    self.split_idxs[split] = 1
  else
    self.split_idxs[split] = idx + 1
  end
	return x, y
end

