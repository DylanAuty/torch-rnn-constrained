# evalTest1.py
# Script to learn/test basic BLEU evaluation capabilities of the python Natural Language Toolkit (nltk)
# To be used with python2.
#
# Dylan Auty, 5/4/2016

from __future__ import division
import nltk, re, pprint
from nltk import word_tokenize
from nltk.translate import bleu_score

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

    # First define some test strings to work with
    refTextRaw = "This is the story of a man who fell from the fiftieth story of a building. While he fell, he reassured himself by repeating, 'So far, so good. So far, so good. So far, so good'. But, the important thing is not the fall - only the landing."
    candidateTextRaw = "This is the story of a man who fell from the 50th floor of a block. To reassure himself while he fell, he repeated, 'So far, so good. So far, so good. So far, so good'. However, the important thing is not the fall. Only the landing."
    refTextTokens = word_tokenize(refTextRaw)
    candidateTextTokens = word_tokenize(candidateTextRaw)

    candidate1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which', 'ensures', 'that', 'the', 'military', 'always', 'obeys', 'the', 'commands', 'of', 'the', 'party']

    candidate2 = ['It', 'is', 'to', 'insure', 'the', 'troops', 'forever', 'hearing', 'the', 'activity', 'guidebook', 'that', 'party', 'direct']

    reference1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'that', 'ensures', 'that', 'the', 'military', 'will', 'forever', 'heed', 'Party', 'commands']

    reference2 = ['It', 'is', 'the', 'guiding', 'principle', 'which', 'guarantees', 'the', 'military', 'forces', 'always', 'being', 'under', 'the', 'command', 'of', 'the', 'Party']

    reference3 = ['It', 'is', 'the', 'practical', 'guide', 'for', 'the', 'army', 'always', 'to', 'heed', 'the', 'directions', 'of', 'the', 'party']

    # Work out the BLEU score
    bleuSc = bleu_score.sentence_bleu([refTextTokens], candidateTextTokens)
    print(bleuSc)


main()  # Python is evil sometimes
