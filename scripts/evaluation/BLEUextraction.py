# BLEUextraction.py
# Script to run every member of a test set through a given model checkpoint,
# then compare the output to the reference in order to calculate the BLEU score.
# It dumps the list of BLEU scores to a file, in a single column.
#
# Expects n arguments:
#   --input_checkpoint /path/to/checkpoint/to/sample.t7
#   --input_data /path/to/test/dataset.json
#   --output_file /path/to/output/file.csv 
#       Note that the output extension is a csv just for forms sake... it's a single column so it's not necessary.
#
# Data JSON is of structure:
# { i : {
#       'data' : "<STRING OF DATA HERE>",
#       'forecast' : "<NL FORECAST HERE>"
#       }
# }
# i is an int, not a string.
#
# Dylan Auty, 29/05/16

import argparse, json
import sys
import shlex
import subprocess
import nltk, re
from nltk import word_tokenize
from nltk.translate import bleu_score


# Use argparse to fetch the input arguments
# Hopefully this is more tabcompletion friendly than optarg
parser = argparse.ArgumentParser()
parser.add_argument('--input_checkpoint')
parser.add_argument('--input_data')
parser.add_argument('--output_file')
args = parser.parse_args() 

if __name__ == '__main__':
    
    # First read in the data and stick it in a dict for easy access.
    data = json.loads(open(args.input_data).read())
    
    print("Writing output to " + args.output_file)

    # Build the command line base strin components.
    comm = 'th sample.lua -gpu -1 -checkpoint ' + args.input_checkpoint + ' -nullstop 1 -start_text '
    #comm = 'th sample.lua -checkpoint ' + args.input_checkpoint + ' -nullstop 1 -start_text '
    
    # Iterate over every item in data, seed data[i]['data'] to the network
    # A note on timing: One iteration of this loop seems to take ~3.9s
    # With 2945 examples, this should take ~3h20m or so.
    # CPU MODE: Single iteration takes ~6.2s => ~5h6m for 2945 examples assuming it doesn't leak memory...
    
    ignored = 0     # Counter for examples ignored due to errors.
    iterationNum = 0   # Iteration counter.
    DataLength = len(data)

    print("Begin BLEU calculation...")
    with open(args.output_file, "a") as outFile:
        for i, ex in data.iteritems():
            print("Iteration: " + `iterationNum`)
            iterationNum += 1
            # Generate the command to run a sample through
            commArgs = shlex.split(comm)
            dataString = ex['data']
            commArgs.append(dataString)
            if len(dataString) == 0:
                print("Data error in iteration " + `i` + ", ignoring...")
                ignored += 1
                break   # Assuming that this means that there's a bad example

            # Sample the model using the seed text.
            try:
                retString = subprocess.check_output(commArgs)
            except:
                print("Sample Error in iteration " + `i` + ", ignoring...")
                ignored += 1
                break   # This means a problem with the sampling.
            else:
                # Process the returned string
                #retString = p.stdout.read()
                retStringSplit = retString.split(dataString, 1) # Split on newline to remove the input argument...
                if len(retStringSplit) < 2:
                    ignored += 1
                    print("Output error: No output seems to be present. Ignoring...")
                    break
                genString = retStringSplit[1]   # genString now contains the sampled forecast
                genString = genString.strip()

                # Now compute BLEU score
                if len(genString) == 0:
                    print("Output error: Returned output is size 0. Ignoring...")
                    ignored += 1
                    break
                genStringToken = word_tokenize(genString)
                refStringToken = word_tokenize(ex['forecast'])
                bleuScore = bleu_score.sentence_bleu([refStringToken], genStringToken)
            
                # Appending to the end of a file
                outFile.write(`bleuScore` + "\n")

    print("Done.")
    print("Ignored Examples: " + `ignored`)
