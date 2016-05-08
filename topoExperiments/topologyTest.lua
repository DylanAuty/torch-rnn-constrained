--[[ topologyTest.lua
-- A way of testing out different network topologies using nngraph
-- Means that the entire network doesn't have to be run every time. 
-- This doesn't test functionality, just a way to experiment with nngraph and
-- produce network diagrams.
-- Dylan Auty, May 2016
--]]

require 'torch'
require 'nn'
require 'nngraph'

-- Define input/hidden/output sizes
ni =1;      -- Input layer size
nh = 8;     -- Hidden layer size
no = 1;     -- Output layer size
len = 13;   -- Number of layers to put together

-- INPUTS
x = nn.Identity()():annotate{
	name = 'INPUT LAYER',
	graphAttributes = {color = 'red'}
}
h0 = nn.Identity()()
--[[
jt = x
jt:annotate{
	name = 'DEBUG: Begin JT'
}
]]--
h = h0
h:annotate{
	name = 'DEBUG: Begin H'
}

-- Define the layers
for i=1, len do
	-- For first layer
	if i==1 then		
		h = nn.Linear(ni+nh, nh)(x):annotate{ -- Each hidden layer is just an nn.Linear for the time being
   		name = 'Dummy LSTM layer (First)',
			graphAttributes = {color = 'blue'}
		}
		--[[
		jt = nn.Identity(1)({h}):annotate{ -- Create a join table for the outgoing skip connections
			name = 'Concatenation node (First)',
			graphAttributes = {color = 'yellow'}
		}
		]]--
	-- For second layer (saves an unnecessary node)
	elseif i==2 then
		h = nn.Linear(ni+nh, nh)(nn.JoinTable(1)({h, x})):annotate{ -- Each hidden layer is just an nn.Linear for the time being
   		name = 'Dummy LSTM layer',
			graphAttributes = {color = 'blue'}
		}
		jt = nn.JoinTable(1)({x, h}):annotate{ -- Create a join table for the outgoing skip connections
			name = 'Concatenation node (First)',
			graphAttributes = {color = 'yellow'}
		}
		
	-- For 2nd layer onwards
	elseif i~=1 and i~=2 then
		
		h = nn.Linear(ni+nh, nh)(nn.JoinTable(1)({h, x})):annotate{ -- Each hidden layer is just an nn.Linear for the time being
   		name = 'Dummy LSTM layer',
			graphAttributes = {color = 'blue'}
		}
		jt = nn.JoinTable(1)({jt, h}):annotate{ -- Create a join table for the outgoing skip connections
			name = 'Concatenation node',
			graphAttributes = {color = 'yellow'}
		}
	end
end

-- Output
y = nn.Linear(nh, no)(jt):annotate{	-- A final linear module for the output.
	name = 'OUTPUT LAYER',
	graphAttributes = {color = 'red'}
}
testNet = nn.gModule({x},{y})	-- Links together the network

-- Generate a graph from the testNet gModule above, and save to file
-- Saves both a .dot and a .svg.
graph.dot(testNet.fg, 'Skip Connection Demonstration', 'SkipConDemoGraph')



