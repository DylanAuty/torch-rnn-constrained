--[[ dimPrint.lua
-- A torch nn module that can be inserted into forward networks.
-- To be used to track the dimensions of tensors in a network.
-- Dylan Auty, May 2016
--]]

require 'torch'
require 'nn'

local DP, parent = torch.class('nn.dimPrint', 'nn.Module')

function DP:__init()
	parent.__init(self)

end

function DP:updateOutput(input)
	print("TYPE: ", type(input))
	print("SIZE: ", #input)
	print(input)
end
