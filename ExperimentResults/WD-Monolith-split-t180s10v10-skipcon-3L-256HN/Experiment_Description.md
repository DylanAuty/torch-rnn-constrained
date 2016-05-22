# WD-Monolith-split-t80s10v10-skipcon-3L-256HN
## Dataset
The base dataset used was the WeatherGov dataset. It was processed into a single, monolithic text file.

The data was paired in the following way:

```
[Raw Data]

[Corresponding forecast]
\0
[Raw Data]

[Corresponding forecast]
\0
```

Note that square brackets denote placeholders here, not literal square brackets. `\0` denotes the null character.

This was split at the character level in the proportions 80/10/10, for training, testing and validation sets respectively.
Note that though this split may occur midway through a data/forecast pair. Due to the large size of the dataset, the effect of the single corrupted example per set on the overall performance will be negligible, therefore this will be ignored for this experiment.

## Network Architecture
The network consists of three hidden layers of 256 LSTM cells each. The connections are feedforward (layer to layer), as well as incoming skip connections (input - every hidden layer) and outgoing skip connections (every hidden layer to the output).

Where a module has multiple inputs to a layer, this is implemented by a concatenation of the input tensors.
The consequence of this is that at the output, the dimensions of this tensor must be adjusted. This is implemented with an nn.Linear module, which applies a linear transformation to a vector input.



