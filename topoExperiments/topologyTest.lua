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

jt = x
h = h0

-- Define the layers
for i=1, len do
    h = nn.Linear(ni+nh, nh)(nn.JoinTable(1)({h, x})):annotate{ -- Each hidden layer is just an nn.Linear for the time being
   	name = 'Dummy LSTM layer',
		graphAttributes = {color = 'blue'}
		}
		jt = nn.JoinTable(1)({jt, h}):annotate{ -- Create a join table for the outgoing skip connections
			name = 'Concatenation node',
			graphAttributes = {color = 'yellow'}
		}
end

-- Output
y = nn.Linear(nh, no)(jt):annotate{	-- A final linear module for the output.
	name = 'OUTPUT LAYER',
	graphAttributes = {color = 'red'}
}
testNet = nn.gModule({h0, x},{y})	-- Links together the network

-- Generate a graph from the testNet gModule above, and save to file
-- Saves both a .dot and a .svg.
graph.dot(testNet.fg, 'Skip Connection Demonstration', 'SkipConDemoGraph')



