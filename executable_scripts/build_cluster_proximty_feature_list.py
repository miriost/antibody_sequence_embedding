# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 10:18:13 2020

@author: mirio
"""
"""
Cluster proximity to feature table

Feture selection
================
Start by feature selection according to the following parameters:
    1. min_number_of_subjects- how many subjects are in each cluster
    2. Significance - only clusters with higher score will be chosen abs(1-score) 
    * We will want to eliminate clusters that appeared already
    
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
import sys, argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--analysis_file_path',
                        help='the filtered data file path')
    parser.add_argument('-ds', '--distances_file_path',
                        help='the distances file path')
    parser.add_argument('-v', '--vectors_file_path',
                        help='the vectors file path')
    parser.add_argument('-of', '--output_folder_path',
                        help='Output folder for the 3 output files - Nearest neighbors file, distances file, and '
                             'results analysis file')
    parser.add_argument('-od', '--output_description',
                        help='description to use inside output file names')
    parser.add_argument('-l', '--label_freq_col', default='HC',
                        help='name of the column in the analysis file with the label frequency')
    parser.add_argument('-S','--significance', help='minimal significance score for cluster selection',
                        default=0.6, type=float)
    parser.add_argument('-m','--min_subjects', help='minimal number of subjects for cluster selection',
                        default=10, type=int)
    args = parser.parse_args()
    
    if not(os.path.isfile(args.analysis_file_path)):
        print('analysis file error, make sure file path exists\nExiting...')
        sys.exit(1)
        date
    if not(os.path.isfile(args.distances_file_path)):
        print('distances file error, make sure file path exists\nExiting...')
        sys.exit(1)
            
    if not(os.path.isfile(args.vectors_file_path)):
        print('vectors file error, make sure file path exists\nExiting...')
        sys.exit(1) 

    # load files    
    inputfile = pd.read_csv(args.analysis_file_path)
    vectors_file = pd.read_csv(args.vectors_file_path)
    distance_file = pd.read_csv(args.distances_file_path)
    
    #load parameters
    min_number_of_subjects = args.min_subjects
    min_significance = args.significance

    #calculate significance score    
    inputfile['SIGNIFICANCE'] = [max(a, 1-a) for a in inputfile[args.label_freq_col]]
    sorted_df = inputfile.sort_values(by=['SIGNIFICANCE'], ascending=False)
    
    #processing
    number_of_features_neto = 0
    number_of_features_bruto = 0
    # A list with the same length as the data, nan on all cells except the chosen features, which will have 1
    selected_feature_indexes = [np.nan] * len(inputfile)
    # A list with the same length as the data, nan on all cells except the chosen points, which will have the index of
    # the original cell which created the cluster
    neighbors_feature_index = [np.nan] * len(inputfile)
    for idx, val in sorted_df.iterrows():
        if val['SIGNIFICANCE'] > min_significance and val['how_many_subjects'] > min_number_of_subjects:
            number_of_features_bruto += 1
            if np.isnan(neighbors_feature_index[idx]):
                number_of_features_neto += 1
                selected_feature_indexes[idx] = 1
                for neighbor in map(int, val['neighbors'][1:-1].split(',')): 
                    neighbors_feature_index[neighbor] = idx

    print('Summary\n=========\n')                     
    print('Significant score: '+str(min_significance))           
    print('Number of features that meet the TH criteria:' + str(number_of_features_bruto))
    print('Number of features after filtration of those appeared as neighbors: ' + str(number_of_features_neto))
    
    # ====
    # build a feature list file, each raw contains feature center, and maximal radius
    c = ['feature_index', 'max_distance']
    c.extend(vectors_file.columns)
    print(c)
    selected_features = np.nonzero(~np.isnan(selected_feature_indexes))[0]
    print('selected features: ', selected_features)
    if len(selected_features) != number_of_features_neto:
        print('Error! selected features number mismatch!!')
    else:
        print(f'Beginning builiding feature list with {number_of_features_neto} features')
    
    features_df = pd.DataFrame(0, index=range(0,number_of_features_neto), columns=c)
    features_df['feature_index'] = selected_features
    features_df['max_distance'] = distance_file.iloc[selected_features].max(axis=1)
    features_df.iloc[:, 2:] = vectors_file.iloc[selected_features]

    features_df.to_csv(os.path.join(args.output_folder_path, args.output_description + '.csv'), index=False)
    print('file saved to ', os.path.join(args.output_folder_path, args.output_description + '.csv'))

    
if __name__ == '__main__':
    main()
