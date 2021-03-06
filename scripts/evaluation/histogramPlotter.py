# histogramPlotter.py
# Input is a file containing a single column of data (going to be using this for BLEU scores)
# Output is a histogram of the data.
#
# Expects 2 arguments:
#   --input_data /path/to/test/dataset.csv
#   --output_file /path/to/output/file.jpg 
#
# Dylan Auty, 31/05/16

import argparse, json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

# Use argparse to fetch the input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--input_data')
parser.add_argument('--output_file')
args = parser.parse_args() 

if __name__ == '__main__':
    
    # Read in the single column csv
    data = np.genfromtxt(args.input_data, delimiter=",")
    
    mean = sum(data) / float(len(data))
    print("Data mean = " + `mean`)
    print("data.mean() = " + `data.mean()`)
    fig1 = plt.figure()
    resultGraph = plt.hist(data, bins=100, color='blue', alpha=0.75)
    plt.axvline(data.mean(), color='b', linestyle='dashed', linewidth=2)
    plt.xlabel("BLEU Score")
    plt.ylabel("Frequency")
    plt.savefig(args.output_file)
    plt.close(fig1)

    print("Done.")

