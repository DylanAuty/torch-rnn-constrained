This experiment was to see if the network could be taught correspondences using its memory alone. For every output character (a character in the forecast), the same data vector was input.

The data vector was produced from the text forecast to cut out human-readable data. It is 47 elements long.

NOTE: This particular run was broken, as the data preprocessing did not batch the forecasts and output up properly.
Instead of data:view(N, -1, T, 47):transpose(1, 2):clone() to calculate the data batches, data:view(-1, N, T, 47):clone() was used. This would not be an issue except the former view method was used to divide up the characters in the forecasts, which could mean that the data was garbage.


