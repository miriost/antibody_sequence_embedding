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
    parser.add_argument('--drop_duplicates', help='Drop duplicated sequences in the same repertoire, (default True)',
                        default=True, type=str2bool)
    parser.add_argument('--n_read_frames', help='how many read frames to use for the embedding, default is all possible'
                                                'read frames', type=int)


    args = parser.parse_args()
    drop_duplicates = args.drop_duplicates
    data_file_path = args.data_file_path
    model_file_path = args.model_file_path
    column = args.column
    n_read_frames = args.n_read_frames

    if not os.path.isfile(data_file_path):
        print('Data file ({}) error! Make sure the file exists and it is *.tsv file.\n'
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
            return list(model.to_vecs(word, n_read_frames=n_read_frames))
        except:
            return np.nan

    print('Generating vectors...')
    vectors = data_file[column].apply(embed_data)

    print('{:.3}% of data not transformed'.format((100*sum(vectors.isna())/data_len)))

    # drop the un translated rows from the file
    data_file = data_file.drop(vectors[vectors.isna()].index, axis=0)
    vectors = vectors[vectors.notna()]

    # save to files:
    data_file_name = os.path.basename(data_file_path).split(".tsv")[0]
    model_file_name = os.path.basename(model_file_path).split(".model")[0]
    dir_name = os.path.dirname(data_file_path)

    data_file_output = os.path.join(dir_name, data_file_name + '_' + model_file_name + '_FILTERED.tsv')
    print('Saving ' + data_file_output)
    data_file.to_csv(data_file_output, sep='\t', index=False)

    vectors = np.array(vectors.tolist())

    vectors_file_output = os.path.join(dir_name, data_file_name + '_' + model_file_name + '_VECTORS.npy')
    print('Saving ' + vectors_file_output)
    np.save(vectors_file_output, vectors)


if __name__ == "__main__":
    main()

