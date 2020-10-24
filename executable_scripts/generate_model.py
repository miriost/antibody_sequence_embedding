#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 15:16:13 2018

@author: miri-o
"""

import sys, os
import time
import pathlib
import argparse
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()).split('antibody_sequence_embedding')[0])
from antibody_sequence_embedding.sequence_modeling import ProtVec

def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', help='an input tsv file with', type=str)
    parser.add_argument('--data_column', help='name of the column with the data for the model generation', type=str)
    parser.add_argument('--corpus_file', help='an input corpus file - if data-file was not provided', type=str)
    parser.add_argument('--desc', help='desc for the output files names', type=str)
    parser.add_argument('--reading_frame', help='reading frame parameter for the prot to vec', type=int)
    parser.add_argument('--trimming', help='start end trimming to the junction', type=int, nargs=2)
    parser.add_argument('--n_gram', help='n-gram parameter from the prot to vec', type=int)
    parser.add_argument('--sample', help='part of data used for creating the model (default 1.0)', type=float,
                        default=1.0)
    parser.add_argument('--seed', help='seed used for sampling the data used for the model', type=int, default=0)

    args = parser.parse_args()

    data = None
    if args.data_file is not None:
        if not os.path.isfile(args.data_file):
            print('Invalid data_file argument: {}\n Existing...'.format(args.data_file))
            sys.exit(2)
        data_df = pd.read_csv(args.data_file, sep='\t')
        if args.data_column not in data_df.columns:
            print('{}} is not in {} columns\n Existing...'.format(args.data_column, args.data_file))
            sys.exit(2)
        data = data_df[args.data_column]

        print('Generated a model from fasta file "{}"\nModel saved to: {}\nCorpus file saved to: {}'.format(
            args.data_file, args.desc + '.model', args.desc + '_corpus.txt'))
    elif args.corpus_file is not None:
        if not os.path.isfile(args.corpus_file):
            print('Invalid corpus_file argument: {}\n Existing...'.format(args.corpus_file))
            sys.exit(2)
    else:
        print('Must provide a data file or a corpus file\n Existing...'.format(args.corpus_file))
        sys.exit(2)

    pv = ProtVec(data=data, corpus=args.corpus_file, n=args.n_gram, reading_frame=args.reading_frame,
                 trim=args.trimming, size=100, out=args.output_file, sg=1, window=5, min_count=2, workers=3,
                 sample=args.sample, random_seed=args.random_seed)

    process_time = (time.time() - t0) / 60  # in minutes
    print('Model built in {} minutes, saving...'.format(process_time))
    pv.save(args.desc + '.model')


if __name__ == "__main__":
    main()
