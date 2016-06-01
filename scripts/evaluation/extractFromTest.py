# extractFromTest.py
# Script to extract individual examples from the test dataset created by preprocess.py.
# Script expects 2 arguments:
#	1) -i /path/to/dataset.h5
#       2) -j /path/to/dataset.json
#       3) -o /path/to/outfile.json
# Dylan Auty, 25/05/2016

import sys
import getopt
import numpy as np
import h5py
import json

def usage():
    sys.exit("Usage: python2 extractFromTest.py [-i /full/path/to/dataset.h5] [-j /full/path/dataset.json] [-o /path/to/outfile.json]")
    return

def main(argv):
    inputFile = ''
    inputJson = ''
    outputJson = ''
    try:
        opts, args = getopt.getopt(argv,":hi:j:o:")
    except:
        usage()
        sys.exit(2)     # Should not get here, should be done in usage()
    for opt, arg in opts:
        if opt == '-h':
            usage()
        elif opt == '-i':
            inputFile = arg
        elif opt == '-j':
            inputJson = arg
        elif opt == '-o':
            outputJson = arg
    
    f = h5py.File(inputFile, "r")
    fjson = json.loads(open(inputJson).read())
    token_to_idx = fjson["token_to_idx"]
    idx_to_token = fjson["idx_to_token"]
    
    testSet = f.get('train')
    testSet = np.array(testSet)
    tString = ''
    outData = []
    outDict = {}
    newlineIdx = token_to_idx["\n"]
    dotIdx = token_to_idx["."]
    
    # Divide up the string while it's still encoded to save time
    startIndex = 0
    i = 0
    outDictIndex = 0
    print("Begin splitting encoded set...")
    while(i+2 < (len(testSet))):   # I hope the python gods don't smite me for this
        # Note that this should inherently ignore the first (probably broken) example unless it happens to have
        # split in exactly the right place - which is a good thing.
        if (testSet[i] == newlineIdx and testSet[i+1] == newlineIdx and testSet[i+2] == dotIdx):    # Look for \n\n in the undecoded string
            outData.append(testSet[startIndex:i])
            i += 2
            outDictIndex += 1
            startIndex = i+1
        else:
            i += 1

    print("Finished splitting encoded set.")
    
    # Note on running speed:
    # ---------------
    # A cursory test has the script up to this point taking 17.368s.
    # Adding in a single decoding of a split using a for loop brings the time to 32.178s.
    # This implies that each split takes roughly 15s, depending on length.
    # With ~2900 examples, this means about 12 hours. Which is rubbish.
    # 
    #   List comprehension gives a time of 17.314s which is ridiculously quick in comparison
    # Further investigation shows that the python string concatenation a += "stringhere" (the original solution)
    # requires copying the entire string each time.
    # This means it runs in O(n^2). List comprehension above *should* run in O(n)... which will save some time.
    # 
    #   Adding in a single iteration of splitting the string again for a forecast and writing to a dict object runs in
    # 16.884s. There is little change from the baseline of ~17s to split the idx array.

    print("Converting encoded set to dictionary of strings...")
    # For each split, convert every idx to a token, write this to an array, then use ''.join to quickly smash it together.
    subsetLoopCounter = 0
    for subset in outData[:-1]: # Note that we need to discard the last split since it's 99% likely to be borked beyond use.
        tString = ''.join([idx_to_token[`subset[i]`] for i in range(0, len(subset))])
        
        # Split the data/forecast pair again and shove it into a dict to be written to a JSON object later.
        tSSplit = tString.split("\n\n", 1)  # Take only the first split - should handle any forecasts with linebreaks.
        
        outDict[subsetLoopCounter] = {}
        outDict[subsetLoopCounter]['data'] = tSSplit[0]
        outDict[subsetLoopCounter]['forecast'] = tSSplit[1]
        subsetLoopCounter += 1

    print("Finished converting. Dumping to JSON.")
    with open(outputJson, 'w') as ofile:
        json.dump(outDict, ofile)

if __name__ == "__main__":
    main(sys.argv[1:])

