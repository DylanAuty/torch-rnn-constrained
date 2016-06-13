# MannWhitneyU.py
# Script to compute the Mann Whitney U statistic and p value of two distributions
# Inputs:
# --input_1 = /path/to/data1.csv
# --input_2 = /path/to/data2.csv
#
# Dylan Auty, 13/06/2016

import argparse
from scipy.stats import *
import numpy as np
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--input_1')
parser.add_argument('--input_2')
args = parser.parse_args()

if __name__ == '__main__':
    # Fetch both distributions into separate numpy arrays
    data1 = np.genfromtxt(args.input_1, delimiter=", ")
    data2 = np.genfromtxt(args.input_2, delimiter=", ")
    
    statistic, pValue = mannwhitneyu(data1, data2)
    
    print("=== INPUT 1 ===")
    print("     input_1 MEAN : " + `data1.mean()`)
    print("     input_1 S.D  : " + `np.std(data1)`)
    print("=== INPUT 2 ===")
    print("     input_2 MEAN : " + `data2.mean()`)
    print("     input_2 S.D  : " + `np.std(data2)`)
    print("=== Test Results ===")
    print("     Mann-Whitney U Statistic : " + `statistic`)
    print("     P-Value                  : " + `pValue`)
    


    

