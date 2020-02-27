#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 17:51:44 2019

@author: miri-o
"""

from Bio import SeqIO
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from itertools import groupby


input_file = '/media/miri-o/Documents/fasta_files/CDR3_from_HCV.fa'


fasta_sequences = SeqIO.parse(open(input_file),'fasta')

#first a histogram of the sequence lengths
#seq_lens = []
#for fasta in fasta_sequences:
#    name, sequence = fasta.id, fasta.seq.tostring()
#    seq_lens.append(len(sequence))
#    
#keys_with_freq = [[len(list(group)), key] for key, group in groupby(sorted(seq_lens))]
#freqs, keys, n = plt.hist(seq_lens, bins = 35)
#plt.title('CDR3 length distribution')

#according to the histogram, 17 is the most frequesnt junction length, i will choose only the ones with length 17
output_file = '/media/miri-o/Documents/fasta_files/CDR3_from_HCV_len19.fa'

# Build a list of short sequences:

short_sequences = [record for record in fasta_sequences if len(record.seq) == 19 ]

print("Found %i short sequences" % len(short_sequences))

SeqIO.write(short_sequences, output_file, "fasta")