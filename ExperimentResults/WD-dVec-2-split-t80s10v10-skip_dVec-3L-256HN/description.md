This experiment did the following:
1. Transform the data string into a 47 element vector of numbers
2. For every character in a given forecast, the assigned data vector was equivalent to the data vector for the entire forecast.
3. Input -> output for the network during training: 
		[data] -> [char1]
		[data] -> [char2]
		etc.

The hope was that the network would be able to handle the input data being the same by the existence of the recurrent topology.

It didn't end up learning properly. Loss remained around 3.

