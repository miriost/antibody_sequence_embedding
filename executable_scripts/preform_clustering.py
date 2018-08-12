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
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', help='a *.csv file where each raw represents an observation')
    parser.add_argument('-m','--method', help='Clustering method, default: kmeans')
    parser.add_argument('-n','--number_of_clusters_per_level', help='Number of clusters for each clustering iteartion. default: 20')
    parser.add_argument('-d','--depth', help='Depth of clustering, default: 2')
    parser.add_argument('-D','--debug_mode', help='Display debug messages, default: False')
    parser.add_argument('-TH','--filtering_TH', help = 'filtering threshold for features uniqeness')
    args = parser.parse_args()
    if not args.method:
        method = 'kmeans'
    else:
        method = args.method
    if not args.number_of_clusters_per_level:
        n = 20
    else:
        n = int(args.number_of_clusters_per_level)
    if not args.depth:
        depth = 2
    else:
        depth = int(args.depth)
    if args.debug_mode:
         debug_mode=args.debug_mode
    else:
        debug_mode = False
    if not args.filtering_TH:
        TH = 100
    else:
        TH = int(args.filtering_TH)
    
    print('Input file for clustering: {}, clustering method: {}, n per layer= {}, depth = {}'.format(args.infile, method, n, depth))
    data = pd.read_csv(args.infile)
    to_fit = data.drop('labels', axis = 1)
#    print(to_fit)
    clust = kmeans.unsupervised_clustering(n, depth, debug_mode, method)
    clust.fit(to_fit)
    clust.visualize()
#    print(clust.clusters_)

#    print(clust.labels_)
    clust.create_feature_table(data['labels'], TH)
    print(clust.feature_table)
    clust.save_feature_table(args.infile, path = os.path.join(os.pardir, os.path.pardir, 'results'))
    
 
#    infile = pd.read_csv(args.infile)
#     filename = args.infile.split('/')[-1]
#    
#    
#    #save to files:
#    path_filtered_files = '/media/miri-o/Documents/Immune2vec/filtered_data_sets/'
#    path_vectors = '/media/miri-o/Documents/Immune2vec/vectors/'
#    infile.to_csv(path_filtered_files+filename[:-4]+'_'+modelname[:-6]+'_FILTERED_DATA.csv')
#    print('Data file saved: ' + path_filtered_files+filename[:-4]+'_'+modelname[:-6]+'_FILTERED_DATA.csv')
#    df.to_csv(path_vectors+filename[:-4]+'_'+modelname[:-6]+'_VECTORS.csv', index = False)
#    print('Vectors file saved: ' + path_vectors+filename[:-4]+'_'+modelname[:-6]+'_VECTORS.csv')

    
if __name__ == "__main__":
   main(sys.argv[1:])   
   
  