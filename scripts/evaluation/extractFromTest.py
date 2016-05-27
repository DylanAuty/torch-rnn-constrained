# extractFromTest.py
# Script to extract individual examples from the test dataset created by preprocess.py.
# Script expects 1 argument:
#	1) -i /path/to/dataset.h5
# Dylan Auty, 25/05/2016

import sys
import getopt
import numpy as np
import h5py
import json
import codecs

def usage():
    sys.exit("Usage: python2 extractFromTest.py [-i /full/path/to/dataset.h5] [-j /full/path/dataset.json]")
    return

def main(argv):
    inputFile = ''
    inputJson = ''
    try:
        opts, args = getopt.getopt(argv,":hi:j:")
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
    
    f = h5py.File(inputFile, "r")
    fjson = json.loads(open(inputJson).read())
    token_to_idx = fjson["token_to_idx"]
    idx_to_token = fjson["idx_to_token"]
    
    testSet = f.get('test')
    arr1 = np.array(testSet)
    tString = ''

    for idx in arr1:
        tString += idx_to_token[str(idx)]
    
    splitArr = tString.split("\n\n")    # Split on every double newline, which denoted a break between forecast and data.
    for i in (1, len(splitArr)):
        if splitArr[i][0] == ".":   # This denotes the beginning of a data chunk. Forecasts do not begin with a ".".
            # panic or something

    # Aim - to find first newline newline dot, trim all above that, chunk data separated by \n\n. (which denotes the end of a previous forecast and the beginning of the next bit of data)

    
    print("NEWLINE: ", token_to_idx["\n"])
    print("newnew: ", token_to_idx["\n"], token_to_idx["\n"], token_to_idx["."])

    #for item in arr1:
    #    if item == "\n":
    #        print("HELLO")


if __name__ == "__main__":
    main(sys.argv[1:])

