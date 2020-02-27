#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 15:16:13 2018

@author: miri-o
"""

import sys, getopt
sys.path.insert(0, "/media/miri-o/Documents")
import time
from miris_tools.sequence_modeling import ProtVec

def main(argv):
   fasta_file = None
   corpus_object = None
   output_corpus_file = 'corpus.txt'
   output_model = 'output_model'
   reading_frame = None
   n_len = 3
   fasta_type = True
   trim_range = None
   try:
      opts, args = getopt.getopt(argv,"hf:c:o:n:r:t:",["fasta-file=","corpus-object=","output=","n=","Reading_frame=","trimming="])
   except getopt.GetoptError:
      print('generate_model.py -f <fasta_file> -c <corpus_object> -o <outputfile> -n <n_length> -r <reading_frame> -t <trimming>')
      sys.exit(2)
   reading_frame = None
   for opt, arg in opts:
      if opt == '-h':
         print('generate_model.py -f <fasta_file> -c <corpus_object> -o <outputfile> -n <n_length> -r <reading_frame> -t <trimming>')
         sys.exit()
      elif opt in ("-f", "--fasta-file"):
          fasta_file = arg
      elif opt in ("-c", "--corpus-object"):
         corpus_object = arg
         fasta_type = False
      elif opt in ("-o", "--output"):
         output_corpus_file = arg
      elif opt in ("-r", "--reading_frame"):
         reading_frame = int(arg)
      elif opt in ("-t", "--trimming"):
         trim_range = arg.replace('(','')
         trim_range = trim_range.replace(')','')
         trim_range = trim_range.split(',')
         trim_range = (int(trim_range[0]), int(trim_range[1]))
      elif opt in ("-n", "--n-gram"):
          n_pharse = arg.replace('(','')
          n_pharse = n_pharse.replace(')','')
          n_pharse = n_pharse.split(',')
          if len(n_pharse) == 1:
              n_len = int(n_pharse[0])
          else:
              n_len = (int(n_pharse[0]), int(n_pharse[1]))
   t0 = time.time()              
   if fasta_type and not fasta_file:
       print('Missing fasta file or corpus file\ngenerate_model.py -f <fasta_file> -c <corpus_object> -o <outputfile> -n <n_length> -r <reading_frame> -t <trimming>')
       sys.exit(2)
   elif fasta_file:
       pv = ProtVec(corpus_fname=fasta_file, corpus=None, n=n_len, reading_frame=reading_frame, trim = trim_range,size=300, out=output_corpus_file, sg=1, window=3, min_count=2, workers=3)
       process_time = (time.time() - t0)/60 # in minutes
       print('Model built in {} minutes, saving...'.format(process_time))
       pv.save(output_corpus_file+ '.model')
       print('generated a model from fasta file "{}"\nModel saved to: {}\nCorpus file saved to: {}'.format(fasta_file, output_corpus_file + '.model', output_corpus_file + '_corpus.txt'))
   elif corpus_object:
       pv = ProtVec(corpus_fname=None, corpus=corpus_object, n=n_len, reading_frame=reading_frame, trim = trim_range, size=300, out=output_corpus_file, sg=1, window=3, min_count=2, workers=3)
       process_time = (time.time() - t0)/60 # in minutes
       print('Model built in {} minutes, saving...'.format(process_time))
       pv.save(output_model + '.model')
       print('generated a model based on corpus object "{}"\nModel saved to: {}\Corpus file saved to: {}'.format(corpus_object, output_corpus_file + '.model', output_corpus_file + '_corpus.txt'))
   
if __name__ == "__main__":
   main(sys.argv[1:])   