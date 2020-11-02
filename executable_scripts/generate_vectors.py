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

    parser.add_argument('--input_file', help='a tsv with the column to vectorize')
    parser.add_argument('--output_column', help='column name for the output (default model file name)')
    parser.add_argument('--column', help='column name to convert to vectors', default="junction")
    parser.add_argument('model', help='a saved word embedding model file')
    parser.add_argument('--size', help='vector size (default 100)', default=100)
    parser.add_argument('--inline', help='Save output on the input file', default=True, type=str2bool)

    args = parser.parse_args()

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

    if args.output_column is None:
        output_column = os.path.basename(args.model).split('.model')[0]
    else:
        output_column = args.output_column

    def embed_data(word):
        try:
            return list(model.to_vecs(word)[0])
        except:
            return np.nan

    print(len(data_file[args.column]))

    data_file[output_column] = data_file[args.column].apply(embed_data)

    print('{:.3}% of data not transformed'.format((100*sum(data_file[output_column] == np.nan)/data_len)))
    
    # drop the un translated rows from the file
    data_file = data_file.drop(data_file.index[data_file[output_column].isnull()])

    # save to files:
    if args.inline is True:
        output_file_name = args.input_file
    else:
        file_name = os.path.basename(args.input_file).split(".tsv")[0]
        dir_name = os.path.dirname(args.input_file)
        output_file_name = os.path.join(dir_name, file_name + '_and_vectors.tsv')

    data_file.to_csv(output_file_name, sep='\t', index=False)
    print('File saved: ' + output_file_name)

    
if __name__ == "__main__":
    main()

