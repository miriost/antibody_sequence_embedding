#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 15:16:13 2018

@author: miri-o
"""

import sys, os
import pathlib
import argparse
import pandas as pd
from embedding.sequence_modeling import ProtVec


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('data_file', help='an input tsv file with', type=str)
    parser.add_argument('desc', help='desc for the output files names', type=str)
    parser.add_argument('--n_dim', help='vector size (default 100)', default=100, type=int)
    parser.add_argument('--n_gram', help='n-gram parameter from the prot to vec (default 3)', type=int, default=3)
    parser.add_argument('--corpus_file', help='an input corpus file - if data-file was not provided', type=str)
    parser.add_argument('--reading_frame', help='reading frame parameter for the prot to vec', type=int)
    parser.add_argument('--sample_fraction', help='part of data used for creating the model (default 1.0)', type=float,
                        default=1.0)
    parser.add_argument('--seed', help='seed used for sampling the data used for the model, (default 0)', type=int,
                        default=0)

    args = parser.parse_args()

    data_column = 'cdr3_aa'

    data = None
    if args.data_file is not None:
        if not os.path.isfile(args.data_file):
            print('Invalid data_file argument: {}\n Existing...'.format(args.data_file))
            sys.exit(2)
        data_df = pd.read_csv(args.data_file, sep='\t')
        if data_column not in data_df.columns:
            print('{}} is not in {} columns\n Existing...'.format(data_column, args.data_file))
            sys.exit(2)
        data = data_df[data_column]
    elif args.corpus_file is not None:
        if not os.path.isfile(args.corpus_file):
            print('Invalid corpus_file argument: {}\n Existing...'.format(args.corpus_file))
            sys.exit(2)
    else:
        print('Must provide a data file or a corpus file\n Existing...'.format(args.corpus_file))
        sys.exit(2)

    pv = ProtVec(data=data, corpus=args.corpus_file, n=args.n_gram, reading_frame=args.reading_frame,
                 size=args.n_dim, out=args.desc, sg=1, window=5, min_count=2, workers=3,
                 sample_fraction=args.sample_fraction, random_seed=args.seed)

    print('Model is ready, saving...')
    pv.save(args.desc + '.model')


if __name__ == "__main__":
    main()

