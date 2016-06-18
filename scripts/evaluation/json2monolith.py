# json2monolith.py
# Script to convert 3 separate JSONs, each of structure:
# { i : {
#       'data' : "<STRING OF DATA HERE>",
#       'forecast' : "<NL FORECAST HERE>"
#       }
# }
# (i is an int, not a string)
# into 1 continuous monoliths of data/forecast pairs. Three splits: train, test, val.
#
# Expects 5 arguments:
#   --input_json /path/to/inputJson.json
#       Note that this is the JSON of the dataset from which the train/test/validation jsons were created.
#       It is needed for encoding/decoding.
#   --input_train /path/to/trainset.json
#   --input_test /path/to/testset.json
#   --input_val /path/to/validationset.json
#   --output_monolith /path/to/outputfile.txt
#
# Dylan Auty, 1/06/16

import argparse, json
import h5py
import sys
import numpy as np

# Use argparse to fetch the input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--input_json')
parser.add_argument('--input_train')
parser.add_argument('--input_test')
parser.add_argument('--input_val')
parser.add_argument('--output_monolith')
args = parser.parse_args() 

if __name__ == '__main__':
    
    # Read the encoder/decoder references into variables
    fjson = json.loads(open(args.input_json).read())
    token_to_idx = fjson["token_to_idx"]
    idx_to_token = fjson["idx_to_token"]

    # Read the data into dicts.
    trainData = json.loads(open(args.input_train).read())
    testData = json.loads(open(args.input_test).read())
    valData = json.loads(open(args.input_val).read())
    
    # Fetch the lengths of the datasets
    trainLength = len(trainData)
    testLength = len(testData)
    valLength = len(valData)

    # Create an HDF5 file and make some groups to put data into
    # HDF5 parses a slash as a subgroup
    # e.g. h.create_dataset(k.strftime('%Y/%m/%d/%H:%M'), data=np.array(v, dtype=np.int8))
    #   should apparently create subgroups, and can be accessed by going h[Y][m][d][H:M].value where all keys are strings
    #       (from here: http://stackoverflow.com/questions/16494669/how-to-store-dictionary-in-hdf5-dataset)

    print("Output will be written to " + args.output_monolith)
    
    # Choose datatype based on the vocabulary size (this taken from scripts/preprocessing.py)
    dtype = np.uint8
    if len(token_to_idx) > 255:
        dtype = np.uint32

    ignored = 0     # Counter for examples ignored due to errors.
    iterationNum = 0   # Iteration counter.
    
    # Note that strings must be encoded before writing.

    with open(args.output_monolith, 'a') as f:
        # Iterate over the training data
        print("Encoding training set...")
        for i, ex in trainData.iteritems(): 
            # i is a unicode string, containing an int. Helpful as a chocolate teapot, that.
            # Modification for dVec data - dataString is already a list, so don't bother encoding it.
            dataString = ex['data']
            forecastString = ex['forecast']
            

            dataLength = len(dataString)
            forecastLength = len(forecastString)
            # Create some numpy arrays to hold the encoded strings
            #dataArr = np.zeros(dataLength, dtype=dtype)
            forecastArr = np.zeros(forecastLength, dtype=dtype)
            
            # Sanity check all the strings
            if len(dataString) == 0:
                print("Data error in iteration " + `i` + ", ignoring...")
                ignored += 1
            
            if len(forecastString) == 0:
                print("Data error in iteration " + `i` + ", ignoring...")
                ignored += 1
            
            # Encode the data and the forecast
            
            # V1: to be used when using strings of forecast data
            #it = 0
            #for char in dataString:
            #    dataArr[it] = token_to_idx[char]
            #    dataArr[it] = char
            #    it += 1

            # V2: to be used when using lists of encoded forecast data
            dataArr = np.asarray(dataString, dtype=dtype)

            it = 0
            for char in forecastString:
                forecastArr[it] = token_to_idx[char]
                it += 1

            # Write to the HDF5
            # Each example is technically a 'dataset'
            h.create_dataset("train/data/" + i, data=dataArr)
            h.create_dataset("train/forecast/" + i, data=forecastArr)

        # Iterate over the test data
        print("Encoding test set...")
        for i, ex in testData.iteritems(): 
            dataString = ex['data']
            forecastString = ex['forecast']

            dataLength = len(dataString)
            forecastLength = len(forecastString)
            #dataArr = np.zeros(dataLength, dtype=dtype)
            forecastArr = np.zeros(forecastLength, dtype=dtype)
            
            if len(dataString) == 0:
                print("Data error in iteration " + `i` + ", ignoring...")
                ignored += 1
            
            if len(forecastString) == 0:
                print("Data error in iteration " + `i` + ", ignoring...")
                ignored += 1
            
            # Encoding
            dataArr = np.asarray(dataString, dtype=dtype)
            
            #it = 0
            #for char in dataString:
            #    dataArr[it] = token_to_idx[char]
            #    it += 1
            
            it = 0
            for char in forecastString:
                forecastArr[it] = token_to_idx[char]
                it += 1

            
            # Write to the HDF5
            h.create_dataset("test/data/" + i, data=dataArr)
            h.create_dataset("test/forecast/" + i, data=forecastArr)

        # Iterate over the validation data
        print("Encoding validation set...")
        for i, ex in valData.iteritems(): 
            dataString = ex['data']
            forecastString = ex['forecast']

            dataLength = len(dataString)
            forecastLength = len(forecastString)
            #dataArr = np.zeros(dataLength, dtype=dtype)
            forecastArr = np.zeros(forecastLength, dtype=dtype)
            
            if len(dataString) == 0:
                print("Data error in iteration " + `i` + ", ignoring...")
                ignored += 1
            
            if len(forecastString) == 0:
                print("Data error in iteration " + `i` + ", ignoring...")
                ignored += 1
            
            # Encoding
            dataArr = np.asarray(dataString, dtype=dtype)
            
            #it = 0
            #for char in dataString:
            #    dataArr[it] = token_to_idx[char]
            #    it += 1
            
            it = 0
            for char in forecastString:
                forecastArr[it] = token_to_idx[char]
                it += 1

            # Write to the HDF5
            # Each example is technically a 'dataset'
            h.create_dataset("val/data/" + i, data=dataArr)
            h.create_dataset("val/forecast/" + i, data=forecastArr)
        
        # Could I have done this in one loop? Yes. Did I? No.

    print("Done.")
    print("Ignored Examples: " + `ignored`)
