#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 10:31:42 2018

@author: miri-o
"""
import pandas as pd
import numpy as np
import os
import sys, argparse
import embedding.sequence_modeling as sequence_modeling


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def main():

    parser = argparse.ArgumentParser(description='embed a column from input file')
    parser.add_argument('input_file', help='a tsv with the column to vectorize')
    parser.add_argument('model', help='a saved word embedding model file')
    parser.add_argument('--column', help='column name to convert to vectors (default cdr3_aa)', default="cdr3_aa")
    parser.add_argument('--n_dim', help='vector size (default 100)', default=100)
    parser.add_argument('--inline', help='Save output on the input file (default True)', default=True, type=str2bool)
    parser.add_argument('--drop_duplicates', help='Drop duplicated sequences in the same repertoire, (default True)',
                        default=True,
                        type=str2bool)

    args = parser.parse_args()
    n_dim = args.n_dim
    drop_duplicates = args.drop_duplicates

    if not os.path.isfile(args.input_file) or args.input_file[:-4] == '.tsv':
        print('Feature file ({}) error! Make sure the file exists and it is *.tsv file.\n'
              'Exiting...'.format(args.input_file))
        sys.exit(1)
    print('Input file for embedding: ', args.input_file, '\nModel: ', args.model)
 
    data_file = pd.read_csv(args.input_file, sep='\t')

    # load saved model
    model = sequence_modeling.load_protvec(args.model)

    # generate a vector for each junction
    data_len = len(data_file)
    print('Data length: ' + str(data_len))

    if drop_duplicates is True:
        data_file.drop_duplicates(subset=['subject.subject_id', args.column], inplace=True)
        print('Data length after dropping duplicates: ' + str(data_len))

    def embed_data(word):
        try:
            return list(model.to_vecs(word))
        except:
            return np.nan

    vectors = pd.DataFrame(data_file[args.column].apply(embed_data), index=data_file.index, columns=['vector'])

    print('{:.3}% of data not transformed'.format((100*sum(vectors['vector'].isna())/data_len)))

    # drop the un translated rows from the file
    vectors = vectors[vectors['vector'].notna()]
    data_file = data_file.drop(vectors.index)

    # save to files:
    file_name = os.path.basename(args.input_file).split(".tsv")[0]
    dir_name = os.path.dirname(args.input_file)

    vectors_file_output = os.path.join(dir_name, file_name + str(n_dim) + 'DIM_VECTORS.npy')

    if args.inline is True:
        data_file_output = args.input_file
    else:
        dir_name = os.path.dirname(args.input_file)
        data_file_output = os.path.join(dir_name, file_name + '_FILTERED.tsv')

    np.save(vectors_file_output, vectors)

    data_file.to_csv(data_file_output, sep='\t', index=False)

    print('Data file saved: ' + data_file_output)
    print('Vectors file saved: ' + vectors_file_output)

    
if __name__ == "__main__":
    main()

