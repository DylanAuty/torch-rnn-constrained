require 'torch'
require 'hdf5'

local utils = require 'util.utils'

local DataLoader = torch.class('DataLoader_dVec')


function DataLoader:__init(kwargs)
  local h5_file = utils.get_kwarg(kwargs, 'input_h5')
  self.batch_size = utils.get_kwarg(kwargs, 'batch_size')
  self.seq_length = utils.get_kwarg(kwargs, 'seq_length')
  --self.con_length = utils.get_kwarg(kwargs, 'con_length')
	self.con_length = 47	-- Counted manually
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
  splits.train = f:read('/train'):all()
  splits.val = f:read('/val'):all()
  splits.test = f:read('/test'):all()
	
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
	trainData = f:read('/train/data'):all()
	trainForecasts = f:read('/train/forecast'):all()
	testData = f:read('/test/data'):all()
	testForecasts = f:read('/test/forecast'):all()
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
	

	print("Sorting the data into sets...")
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
	splits.train.data = trainData
	splits.train.forecasts = trainForecasts
	splits.test.data = testData
	splits.test.forecasts = testForecasts
	splits.val.data = valData
	splits.val.forecasts = valForecasts

  self.x_splits = {}
	self.y_splits = {}
  self.split_sizes = {}

	-- Aim of this is to construct a set of tuples of [dataVec][forecastChar]
	-- At the end of this, the number of these can be counted and batches made
	-- Allows sensibly sized batches this way...

	-- This loop constructs minibatches ready for use
  -- To make life simpler, aim is to keep this loop with the same output:
	-- 	Just x_splits, y_splits, split_sizes.
	-- All constraint stuff should be sorted out in here.
	
	--[[
	for split, set in pairs(splits) do -- For every top level dataset split (train/test/val)
    -- Split is one of 3 sets of datasets
		local firstFlag = 0
		print("Reshaping data in split:", split)

		local split_x = {}
		local split_y = {}
		local vx = torch.ByteTensor()
		local vy = torch.ByteTensor()
		local vvx = torch.ByteTensor()
		local vvy = torch.ByteTensor()
		local vvchars = torch.ByteTensor()
		local vvdata = torch.ByteTensor()

		-- First, for every example, we append to x and y.
		for datasetNum, v in pairs(set.forecasts) do
			--print(#set.data[datasetNum])

			-- Dimensions now:
			-- 	v (forecast): (L) (where L is the length of the forecast)
			-- 	set.data[datasetNum]: (L, 47)
	
			if firstFlag == 0 then
				vx = set.data[datasetNum]:clone()	
				vy = v:clone()
				firstFlag = 1
			else
				vx = torch.cat(vx, set.data[datasetNum], 1)
				vy = torch.cat(vy, v, 1)
			end
		end
		
		
		-- Now we slice according to batch size and return
		local num = vy:nElement()
		local extra = num % (N * T)
		
		-- If L is number of samples total and B is number of batches produced,
		-- then for data we need to do (L, 47) -> (B, N, T, 47)
		-- 			for forecasts we need to do (L) -> (B, N, T)
		-- Want data to then have a forecast character appended to the end
		-- -- MODIFICATON
		-- Concatenate characters to the end of the batches
		-- (B, N, T, 47) + (B, N, T)
		-- Add singleton dimension to characters (B, N, T) -> (B, N, T, 1)
		-- Concatenate along the 4th dimension.

		vvdata = vx[{{1, num - extra}, {}}]:view(N, -1, T, 47):transpose(1, 2):clone()
		vvchars = vy[{{1, num - extra}}]:view(N, -1, T):transpose(1, 2):clone()
	
		vvcharsUnsqueezed = torch.ByteTensor(vvchars:size(1), vvchars:size(2), vvchars:size(3), 1)
		vvcharsUnsqueezed[{{}, {}, {}, 1}] = vvchars -- Unsqueezing isn't fun with ByteTensors
		vvx = torch.cat(vvdata, vvcharsUnsqueezed, 4)
		vvy = vy[{{2, num - extra + 1}}]:view(N, -1, T):transpose(1, 2):clone()
		
		self.x_splits[split] = vvx
		self.y_splits[split] = vvy
		self.split_sizes[split] = vvx:size(1)
	end
	
	
	torch.save("charByChar_dVec_x_splits", self.x_splits)
	torch.save("charByChar_dVec_y_splits", self.y_splits)
	torch.save("charByChar_dVec_split_sizes", self.split_sizes)
	--]]
	
	print("Loading files...")
	print("Loading x_splits")
	self.x_splits = torch.load("charByChar_dVec_x_splits")
	print("Loading y_splits")
	self.y_splits = torch.load("charByChar_dVec_y_splits")
	print("Loading split_sizes")
	self.split_sizes = torch.load("charByChar_dVec_split_sizes")
	print("Finished loading Torch objects from file")
	
  
	self.split_idxs = {train=1, val=1, test=1}
end


function DataLoader:nextBatch(split)
	-- y will be the same as the input but shifted by one
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

