#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 10:31:42 2018

@author: miri-o
"""
import numpy as np
import os
import sys, argparse
from sklearn.decomposition import PCA


def main():

    parser = argparse.ArgumentParser(description='create a new vectors file after pca dimension reduction')
    parser.add_argument('vectors_file_path', help='a npy vectors file for the pca dimension reduction')
    parser.add_argument('--n_dim', help='number of the dimensions to reduce to', default=2, type=int)

    args = parser.parse_args()

    n_dim = args.n_dim
    vectors_file_path = args.vectors_file_path

    if not os.path.isfile(vectors_file_path) or vectors_file_path[:-4] == '.npy':
        print('Vectors file ({}) error! Make sure the file exists and it is *.npy file.\n'
              'Exiting...'.format(vectors_file_path))
        sys.exit(1)

    vectors_file = np.load(vectors_file_path)

    # save to files:
    file_name = os.path.basename(vectors_file_path).split("VECTORS.npy")[0]
    dir_name = os.path.dirname(vectors_file_path)

    output_file_path = os.path.join(dir_name, file_name + '_' + str(n_dim) + 'DIM_PCA_VECTORS.npy')

    print('Reducing dimension {}->{} using PCA...'.format(vectors_file.shape[1], n_dim))
    pca = PCA(n_components=n_dim)
    vectors = pca.fit_transform(vectors_file)

    print('Saving ' + output_file_path)
    np.save(output_file_path, vectors)


if __name__ == "__main__":
    main()

