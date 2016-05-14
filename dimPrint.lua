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
	self.output = input
	return self.output
end

function DP:updateGradInput(input, gradOutput)
	self.gradInput = gradOutput
print("BACKWARDS: ", self.msg)
	print("TYPE: ", type(self.gradInput))
	print("SIZE: ")
	print(#self.gradInput)
	if(type(self.gradInput) == 'table') then
		print(self.gradInput)
	end
	return self.gradInput
end

function DP:clearState()		-- This function lifted from nn.Identity()
   -- don't call set because it might reset referenced tensors
   local function clear(f)
      if self[f] then
         if torch.isTensor(self[f]) then
            self[f] = self[f].new()
         elseif type(self[f]) == 'table' then
            self[f] = {}
         else
            self[f] = nil 
         end
      end 
   end 
   clear('output')
   clear('gradInput')
   return self
end

--[[
function DP:backward(input)
	print("BACKWARDS: ", self.msg)
	print("TYPE: ", type(input))
	print("SIZE: ")
	print(#input)
	if(type(input) == 'table') then
		print(input)
	end
	return input
end
]]--
