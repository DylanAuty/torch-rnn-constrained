# fileBLEU.py
# Script to take as input two text files, a candidate and a reference, then return their BLEU score.
# To be used with python2.
#
# USAGE: python fileBLEU.py candidateFile.txt ReferenceFile.txt
# This script currently only supports one reference file.
#
# Dylan Auty, 6/4/2016

from __future__ import division
import nltk, re, pprint
from nltk import word_tokenize
from nltk.translate import bleu_score
import sys

def main():
    """ 
    bleu function parameters:
        bleu(candidate, references, weights)  
        :param candidate: a candidate sentence
        :type candidate: list(str)
        :param references: reference sentences
        :type references: list(list(str))
        :param weights: weights for unigrams, bigrams, trigrams and so on
        :type weights: list(float) 
    """
    
    # Command line argument checking
    if(len(sys.argv) != 3):
            sys.exit("ERROR: Invalid number of arguments, expecting 2")

    # Import the files, first the candidate into cFile and the reference to rFile
    cFile = open(sys.argv[1])
    rFile = open(sys.argv[2])
    
    cRaw = cFile.read()
    rRaw = rFile.read()

    # Then tokenize them both
    cToken = word_tokenize(cRaw)
    rToken = word_tokenize(rRaw)

    # Finally compute the BLEU score
    
    bleuSc = bleu_score.sentence_bleu([rToken], cToken)
    print(bleuSc)


main()
