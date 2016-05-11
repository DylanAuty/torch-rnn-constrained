--[[ dimPrint.lua
-- A torch nn module that can be inserted into forward networks.
-- To be used to track the dimensions of tensors in a network.
-- Dylan Auty, May 2016
--]]

require 'torch'
require 'nn'

local DP, parent = torch.class('nn.dimPrint', 'nn.Module')

function DP:__init(msg)
	parent.__init(self)
	self.msg = msg
end

function DP:updateOutput(input)
	print(self.msg)
	print("TYPE: ", type(input))
	print("SIZE: ")
	print(#input)
	if(type(input) == 'table') then
		print(input)
	end
	return input
end
