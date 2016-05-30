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
    
    # Build the command line base strin components.
    comm = 'th sample.lua -checkpoint ' + args.input_checkpoint + ' -nullstop 1 -start_text '

    # Iterate over every item in data, seed data[i]['data'] to the network
    print("Begin BLEU calculation...")
    with open(args.output_file, "a") as f:
        for i, ex in data.iteritems():
            # Generate the command to run a sample through
            commArgs = shlex.split(comm)
            dataString = ex['data']
            commArgs.append(dataString)
            
            # Sample the model using the seed text.
            p = subprocess.Popen(commArgs, stdout=subprocess.PIPE)
            
            # Process the returned string
            retString = p.stdout.read()
            retStringSplit = retString.split(dataString, 1) # Split on newline to remove the input argument...
            print(retStringSplit[1])
            sys.exit()
            genString = retStringSplit[1]   # genString now contains the sampled forecast
            genString = genString.strip()

            # Now compute BLEU score
            genStringToken = word_tokenize(genString)
            refStringToken = word_tokenize(ex['forecast'])
            bleuScore = bleu_score.sentence_bleu([refStringToken], genStringToken)
            
            # Appending to the end of a file
            outFile.write(`bleuScore`)
    
    print("Done.")

