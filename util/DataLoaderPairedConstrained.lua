require 'torch'
require 'hdf5'

local utils = require 'util.utils'

local DataLoader = torch.class('DataLoaderPairedConstrained')


function DataLoader:__init(kwargs)
  local h5_file = utils.get_kwarg(kwargs, 'input_h5')
  self.batch_size = utils.get_kwarg(kwargs, 'batch_size')
  self.seq_length = utils.get_kwarg(kwargs, 'seq_length')
  local N, T = self.batch_size, self.seq_length
	
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
	
	trainData = f:read('/train/data'):all()
	trainForecasts = f:read('/train/forecast'):all()
	testData = f:read('/test/data'):all()
	trainForecasts = f:read('/test/forecast'):all()
	valData = f:read('/val/data'):all()
	valForecasts = f:read('/val/forecast'):all()

	-- originally, splits contained 3 datasets - train, test, val
	-- Now it contains 3 groups of 2 groups each of many datasets.
	
	-- The splits
	splits.train = {}
	splits.train.data = {}
	splits.train.forecasts = {}
	splits.test = {}
	splits.test.data = {}
	splits.test.forecasts = {}
	splits.val = {}
	splits.val.data = {}
	splits.val.forecasts = {}
	
	-- Fill the sets
	for key, item in pairs(trainData) do
		splits.train.data[key] = item
		splits.train.forecasts[key] = trainForecasts[key]
	end
	for key, item in pairs(testData) do
		splits.test.data[key] = item
		splits.test.forecasts[key] = testForecasts[key]
	end
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
		-- v is the array of characters within the given split

		for datasetNum, v in pairs(set.forecasts) do
			local num = v:nElement()
    	local extra = num % (N * T)
			-- num is number of chars in the split
			-- N is batch length
			-- T is sequence length

    	-- Chop out the extra bits at the end to make it evenly divide
			-- vx is of dimension (V, N, T)
			-- 	Need to make it (V + C, N, T) where C is constraint size
			local vx = v[{{1, num - extra}}]:view(N, -1, T):transpose(1, 2):clone()	
			local vy = v[{{2, num - extra + 1}}]:view(N, -1, T):transpose(1, 2):clone()
			
			-- Now extract and append the constraint vector in the right place...
			vc = set.data[datasetNum]
			

    	-- x and y are the input and reference output respectively
			-- x will have the constraint set concatenated onto it. Always the same for every x batch.
			-- The constraint set will be chopped off when the network receives it (I hope)
			-- y is the same as x, just transposed by 1
			-- Note that the outputs are indexed by split alone.
			self.x_splits[split] = vx
    	self.y_splits[split] = vy
    	self.split_sizes[split] = vx:size(1)
  	end
	end

  self.split_idxs = {train=1, val=1, test=1}
end


function DataLoader:nextBatch(split)
	-- To be changed to: construct a batch, concatenate the constraint vector to the end, return that back.
	-- Also y will be the same (the input but shifted by one)
  local idx = self.split_idxs[split]
  assert(idx, 'invalid split ' .. split)
  local x = self.x_splits[split][idx]
  local y = self.y_splits[split][idx]
  if idx == self.split_sizes[split] then
    self.split_idxs[split] = 1
  else
    self.split_idxs[split] = idx + 1
  end
  return x, y
end

