# July 2017
# Winston Becker
# This script counts how many times a list of known sequences occur in a fastq file.

###############################
###############################
# Load in Libraries
import scipy
import matplotlib
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
import sys
import os
import seaborn as sns
import time
import argparse
from string import maketrans

###############################
###############################
revCompTrans = maketrans("ACGT", "TGCA")

###############################
###############################
# Define Functions
def countSequences(SequenceDataOnly, SequenceDataOnly_R2, sequenceList, includeR2 = True):
	initialCounts = [0]*len(sequenceList)
	countDict = dict(zip(sequenceList, initialCounts))

	printcounter = 0
	percentComplete = 0
	numSequences = len(SequenceDataOnly)
	k=0
	m=0
	for i in range(len(SequenceDataOnly)):
		for j in [36,35,34,33,32,31,30,29,28,27]:
			if SequenceDataOnly[i][0:j] in countDict:
				k = k+1
				if includeR2:
					if revComp(SequenceDataOnly[i][0:j]) in SequenceDataOnly_R2[i]:
						m = m+1
						countDict[SequenceDataOnly[i][0:j]] = countDict[SequenceDataOnly[i][0:j]] +1
						break
				else:
					countDict[SequenceDataOnly[i][0:j]] = countDict[SequenceDataOnly[i][0:j]] +1
	print "Total sequences in read 1 file: "+ str(len(SequenceDataOnly))
	print "Total read 1 sequences present in variant table: "+ str(k)
	print "Total read 1 sequences present in variant table that were also in read 2: "+ str(m)
	counts = pd.DataFrame(countDict.items())
	counts.columns = ['sequence', 'counts']
	return counts

def revComp(sequence):
	# Reverse complement a sequence
	return sequence.translate(revCompTrans)[::-1]

###############################
###############################
# Main

# User defined names
fastqLocations = './miR_21_fastqs'
variantTableFilename = './miR21_Ago_variant_table.txt'
includeR2 = True # True if you want to check if the sequence occurs correctly in R1 and R2

# Iterate through fastq files
fileNames = []
for file in os.listdir(fastqLocations):
	if file.endswith("R1_001.fastq"):
		fileNames.append(os.path.join(fastqLocations, file))

VariantTable = pd.DataFrame.from_csv(variantTableFilename, sep='\t', header=None, index_col=0)
sequenceList = list(VariantTable.index)

for name in fileNames:
	print "Filename: " +str(name)
	seq_data = pd.read_table(name, header=None)
	seq_data_R2 = pd.read_table(name[0:-12]+"R2_001.fastq", header=None)
	SequenceDataOnly = list(seq_data.iloc[1::4, :][0])
	SequenceDataOnly_R2 = list(seq_data_R2.iloc[1::4, :][0])
	countTable = countSequences(SequenceDataOnly, SequenceDataOnly_R2, sequenceList, includeR2)
	countTable.to_csv(name[0:-12] + 'counts.tsv', sep="\t", index = False)










