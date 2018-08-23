#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 15:18:09 2018

@author: miri-o
"""

import sys, argparse
import os
sys.path.insert(0, os.path.join(os.pardir, os.path.pardir))
from miris_tools import kmeans
import pandas as pd
import numpy as np

def main(argv):
    parser=argparse.ArgumentParser(
        description='''perform_clustering.py function performs clustering on an N-dimensional data, and produces a feature table where each row is an observaion and each column is a feature (a cluster), the value of each cell in the table is the frequency of the observation for the specific feature''',
        epilog="""All's well that ends well.""")

    parser.add_argument('vector_file', help='a *.csv file where each raw is a vector represents an observation')
    parser.add_argument('labels_file', help='a *.csv file with the same number of rows as the vector file, containing the observation label for each row')
    parser.add_argument('-m','--method', help='Clustering method, default: kmeans', default = 'kmeans')
    parser.add_argument('-n','--number_of_clusters_per_level', help='Number of clusters for each clustering iteartion. default: 20', type = int, default = 20)
    parser.add_argument('-d','--depth', help='Depth of clustering, default: 2', type = int, default = 2)
    parser.add_argument('-D','--debug_mode', help='Display debug messages, default: False', default = False)
    parser.add_argument('-TH','--filtering_TH', help = 'filtering threshold for features uniqeness.\nExpeceted values: 0-100, default: 100', type = int, default = 100)
    parser.add_argument('-I', '--ingore_columns', nargs = '+', help = 'A list of columns in vector file to ignore for clustering, default: []')
    parser.add_argument('-l', '--label_col_name', help = 'The column name in labels file, default: labels', default = 'labels')
    parser.add_argument('-v', '--visualize', help = 'Visualize clustering result, default: False', default = False)

        
    args = parser.parse_args()
    n = args.number_of_clusters_per_level
    depth = args.depth
    debug_mode=args.debug_mode
    TH = args.filtering_TH

    print('Input file for clustering: {}, clustering method: {}, n per layer= {}, depth = {}'.format(args.vector_file, args.method, n, depth))
    if not os.path.isfile(args.vector_file) or args.vector_file[:-4] == '.csv':
        print('Input file file error! Make sure the file exists and it is *.csv file.\nExiting...')
        sys.exit(1)
    vectors = pd.read_csv(args.vector_file)
    if args.ingore_columns:
        to_fit = vectors.drop(args.ignore_columns, axis = 1)
    else:
        to_fit = vectors.copy()
    if not os.path.isfile(args.labels_file) or args.labels_file[:-4] == '.csv':
        print('Input file file error! Make sure the file exists and it is *.csv file.\nExiting...')
        sys.exit(1)
    else:
        if os.path.samefile(args.labels_file,args.vector_file):
            if args.label_col_name in vectors.columns:
                labels = vectors[args.label_col_name]
            else:
                print(args.label_col_name + " column doesn't exist.\nExiting...")
                sys.exit(1)
        else:
            labels_file = pd.read_csv(args.labels_file)
            if args.label_col_name in labels_file.columns:
                labels = labels_file[args.label_col_name]
            else:
                print(args.label_col_name + " column doesn't exist.\nExiting...")
                sys.exit(1)
    if len(labels) != len(vectors):
        print('Vectors and labels length mismatch. Vectors length: {}, labels length: {}\nExiting...'.format(len(vectors), len(labels)))
        sys.exit(1)
        
    clust = kmeans.unsupervised_clustering(n, depth, debug_mode, args.method)
    clust.fit(to_fit)
    if args.visualize:
        clust.visualize()     
    clust.create_feature_table(labels, TH)
    print('Clustering finished\n [ {} rows x {} columns ]'.format(clust.feature_table.shape[0], clust.feature_table.shape[1]))
    #print(clust.feature_table)
    feature_table_path = os.path.join(os.pardir, os.path.pardir, 'feature_table')
    if not os.path.exists(feature_table_path):
        os.mkdir(feature_table_path)        
        
    clust.save_feature_table(args.vector_file, path = feature_table_path)
    
 
#    vector_file = pd.read_csv(args.vector_file)
#     filename = args.vector_file.split('/')[-1]
#    
#    
#    #save to files:
#    path_filtered_files = '/media/miri-o/Documents/Immune2vec/filtered_data_sets/'
#    path_vectors = '/media/miri-o/Documents/Immune2vec/vectors/'
#    vector_file.to_csv(path_filtered_files+filename[:-4]+'_'+modelname[:-6]+'_FILTERED_DATA.csv')
#    print('Data file saved: ' + path_filtered_files+filename[:-4]+'_'+modelname[:-6]+'_FILTERED_DATA.csv')
#    df.to_csv(path_vectors+filename[:-4]+'_'+modelname[:-6]+'_VECTORS.csv', index = False)
#    print('Vectors file saved: ' + path_vectors+filename[:-4]+'_'+modelname[:-6]+'_VECTORS.csv')

    
if __name__ == "__main__":
   main(sys.argv[1:])   
   
  