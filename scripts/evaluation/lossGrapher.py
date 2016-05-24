# lossGrapher.py
# Script to generate graphs from logfile2csv.py 
# Input data is in 2 files:
#   1) checkpointLoss.csv
#   2) trainProgress.csv
#
# checkpointLoss.csv has validation loss data:
#   Epoch, iteration, val_loss
# trainProgress.csv has iteration by iteration loss data.
#   Epoch, iteration, loss
#
# Takes 2 arguments:
#   -i /path/to/input/dir/
#   -o /path/to/output/dir/
# 
# Dylan Auty, 22/05/16

import sys
import getopt
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import StringIO
import collections

plt.ioff()

def usage():
    sys.exit("Usage: python2 lossGrapher.py -i /path/to/input/dir -o /path/to/output/dir")
    return

def main(argv):
    CLin = ''
    TPin = ''
    CLout1 = ''
    CLout2 = ''
    TPout1 = ''
    TPout2 = ''
    TPout3 = ''

    try:
        opts, args = getopt.getopt(argv, ":hi:o:")
    except:
        usage()
        sys.exit(2) # This should just... never happen

    for opt, arg in opts:
        if opt == '-h':
            usage()
            sys.exit(2)
        elif opt == '-i':
            CLin = str(arg) + "checkpointLoss.csv"
            TPin = str(arg) + "trainProgress.csv"
        elif opt == '-o':
            CLout1 = str(arg) + "Epoch_valLoss.jpg"
            CLout2 = str(arg) + "Iteration_valLoss.jpg"
            TPout1 = str(arg) + "Epoch_trainLoss.jpg"
            TPout2 = str(arg) + "Iteration_trainLoss.jpg"
            TPout3 = str(arg) + "Epoch_aveTrainLoss.jpg"
        else:
            usage()
            sys.exit(2)

    # Read data from .csv files
    CLdata = np.genfromtxt(CLin, delimiter=',', names=['Epoch', 'Iteration', 'Validation_Loss'])
    TPdata = np.genfromtxt(TPin, delimiter=',', names=['Epoch', 'Iteration', 'Training_Loss'])
    TPave = []
    # Computing average training loss per epoch
    d = collections.defaultdict(list)
    for item in TPdata:
        d[item[0]].append(item)
    for key, value in d.items():
        aveTrainLoss = sum(value[2])/float(len(value))
        TPave.append([key, aveTrainLoss])
    
    TPave = np.asarray(TPave)

    # Graphs:
    # Epoch vs. validation loss
    fig1 = plt.figure()
    plt.plot(CLdata['Epoch'], CLdata['Validation_Loss'], color='b', label='Training Progress, Epoch/Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.savefig(CLout1)
    plt.close(fig1)

    # Iteration vs. validation loss
    fig2 = plt.figure()
    plt.plot(CLdata['Iteration'], CLdata['Validation_Loss'], color='b', label='Training Progress, Iteration/Validation Loss')
    plt.xlabel("Iteration")
    plt.ylabel("Validation Loss")
    plt.savefig(CLout2)
    plt.close(fig2)
    
    # Epoch vs. training loss
    fig3 = plt.figure()
    plt.plot(TPdata['Epoch'], TPdata['Training_Loss'], color='b', label='Training Progress, Epoch/Training Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.savefig(TPout1)
    plt.close(fig3)
    
    # Epoch vs. ave. training loss
    fig4 = plt.figure()
    plt.plot(TPave[:,0], TPave[:,1], color='b', label='Training Progress, Epoch/Average Training Loss')
    plt.xlabel("Iteration")
    plt.ylabel("Average Training Loss")
    plt.savefig(TPout3)
    plt.close(fig4)
    
    # Iteration vs. training loss
    fig5 = plt.figure()
    plt.plot(TPdata['Iteration'], TPdata['Training_Loss'], color='b', label='Training Progress, Iteration/Training Loss')
    plt.xlabel("Iteration")
    plt.ylabel("Training Loss")
    plt.savefig(TPout2)
    plt.close(fig5)

if __name__ == "__main__":
    main(sys.argv[1:])


