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
    parser.add_argument('data_file_path', help='a tsv with the column to vectorize')
    parser.add_argument('model_file_path', help='a saved word embedding model file')
    parser.add_argument('--column', help='column name to convert to vectors (default cdr3_aa)', default="cdr3_aa")
    parser.add_argument('--n_dim', help='vector size (default 100)', default=100)
    parser.add_argument('--inline', help='Save output on the input file (default True)', default=True, type=str2bool)
    parser.add_argument('--drop_duplicates', help='Drop duplicated sequences in the same repertoire, (default True)',
                        default=True,
                        type=str2bool)

    args = parser.parse_args()
    n_dim = args.n_dim
    drop_duplicates = args.drop_duplicates
    data_file_path = args.data_file_path
    model_file_path = args.model_file_path
    column = args.column
    inline = args.inline

    if not os.path.isfile(data_file_path) or data_file_path[:-4] == '.tsv':
        print('Feature file ({}) error! Make sure the file exists and it is *.tsv file.\n'
              'Exiting...'.format(data_file_path))
        sys.exit(1)
    print('Input file for embedding: ', data_file_path, '\nModel: ', model_file_path)
 
    data_file = pd.read_csv(data_file_path, sep='\t')

    # load saved model
    model = sequence_modeling.load_protvec(model_file_path)

    # generate a vector for each junction
    data_len = len(data_file)
    print('Data length: ' + str(data_len))

    if drop_duplicates is True:
        data_file.drop_duplicates(subset=['subject.subject_id', column], inplace=True)
        print('Data length after dropping duplicates: ' + str(data_len))

    def embed_data(word):
        try:
            return list(model.to_vecs(word))
        except:
            return np.nan

    vectors = data_file[column].apply(embed_data)

    print('{:.3}% of data not transformed'.format((100*sum(vectors.isna())/data_len)))

    # drop the un translated rows from the file
    vectors = vectors[vectors.notna()]
    data_file = data_file.drop(vectors[vectors.isna()].index, axis=0)

    # save to files:
    file_name = os.path.basename(data_file_path).split(".tsv")[0]
    dir_name = os.path.dirname(data_file_path)

    vectors_file_output = os.path.join(dir_name, file_name + '_' + str(n_dim) + 'DIM_VECTORS.npy')

    if inline is True:
        data_file_output = data_file_path
    else:
        dir_name = os.path.dirname(data_file_path)
        data_file_output = os.path.join(dir_name, file_name + '_FILTERED.tsv')

    vectors = np.array(vectors.tolist())
    np.save(vectors_file_output, vectors)

    data_file.to_csv(data_file_output, sep='\t', index=False)

    print('Data file saved: ' + data_file_output)
    print('Vectors file saved: ' + vectors_file_output)

    
if __name__ == "__main__":
    main()

