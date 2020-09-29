# -*- coding: utf-8 -*-
#!/bin/sh
''''exec python -u -- "$0" ${1+"$@"} # '''

"""
Created on Thu Apr 23 10:18:13 2020

@author: mirio
"""
"""
Cluster proximity to feature table

Feture generation script 
================
Input:
Feature list - a file containing the relevant clusters (max distance + center)
data file
vectors file

Output:
    Feature file, where each row is a subject, each column is a sequence, each cell is the normlized frequency of sequences in each feature
    
"""

import pandas as pd
import numpy as np
import sys, argparse
import os
import random
import ray
import time
from datetime import datetime
from sklearn.metrics.pairwise import pairwise_distances

ray.init(memory=20*1024*1024*1024, object_store_memory=20*1024*1024*1024)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--features_list',
                        help='feature list file, contains the list of relevent features, including feature '
                             'center and maximal distance from it')
    parser.add_argument('-d', '--data_file_path',  help='the filtered data file path')
    parser.add_argument('-v', '--vectors_file_path',  help='the vectors file path')
    parser.add_argument('-of', '--output_folder_path', default="./",  help='Output folder for the feature table')
    parser.add_argument('-od', '--output_description',  help='description to use inside output file names')
    parser.add_argument('-s', '--subject_col_name',
                        help='subject column name in data file, default "FILENAME"', default='FILENAME', type=str)
    parser.add_argument('-l', '--labels_col_name',
                        help='labels column name in data file, default "labels"', default='labels', type=str)
    parser.add_argument('-c', '--cpus',
                        help='number of cpus to run parallel computing', default=2, type=int)
    args = parser.parse_args()
    
    if not(os.path.isfile(args.features_list)):
        print('feature list file error, make sure file path exists\nExiting...')
        sys.exit(1)
                
    if not(os.path.isfile(args.data_file_path)):
        print('feature file error, make sure file path exists\nExiting...')
        sys.exit(1)

    if not os.path.isfile(args.vectors_file_path):
        print('vectors file error, make sure file path exists\nExiting...')
        sys.exit(1) 

    # load files
    cpus = args.cpus
    feature_list = pd.read_csv(args.features_list, index_col=0)
    data_file = pd.read_csv(args.data_file_path, sep='\t')
    vectors_file = pd.read_csv(args.vectors_file_path)

    if args.labels_col_name not in data_file.columns:
        print(f'label "{args.labels_col_name}" column name doesnt exist in data file.\nExiting...')
        sys.exit(1)
        
    if args.subject_col_name not in data_file.columns:
        print(f'"{args.subject_col_name}" column name doesnt exist in data file.\nExiting...')
        sys.exit(1)
    
    if 'feature_index' not in feature_list.columns:
        print(f'"feature list file error, no "feature index" column. please check.\n exiting...')
        sys.exit(1)
    else:
        print(f'feature indexes: {feature_list.index}')

    by_subject = data_file.groupby(args.subject_col_name)
    # for each subject - add feature frequency columns
    result_ids = []
    for subject, frame in by_subject:
        subject_vectors = vectors_file.iloc[frame.index]
        result_ids += [get_subject_feature_table.remote(subject, feature_list, subject_vectors, cpus)]
    features_table = pd.concat([ray.get(res_id) for res_id in result_ids])

    # add subject labels column
    features_table[args.labels_col_name] = None
    for subject, frame in by_subject:
        label = frame[args.labels_col_name].unique()[0]
        features_table.loc[subject, args.labels_col_name] = label

    # normalize the feature table
    normalized_features_table = features_table.div(features_table.sum(axis=1), axis=0)
    # save to file
    normalized_features_table.to_csv(os.path.join(args.output_folder_path, args.output_description +
                                                  '_feature_table.csv'))
      
    print('file saved to ', os.path.join(args.output_folder_path, args.output_description + '_feature_table.csv'))


@ray.remote
def get_subject_feature_table(subject, feature_list, subject_vectors, cpus=2):
    # create an empty matrix, each raw is a subject, each column is a feature (cluster)
    features = feature_list.iloc[:, -100:]
    max_distance = feature_list.loc[:, 'max_distance']

    print('Start creating feature table for subject {}'.format(subject))
    t0 = time.time()
    subject_features_table = pd.DataFrame(0, index=[subject], columns=feature_list['feature_index'])
    total_feature_count = 0
    # for each vector belonging to the subject
    for idx, vector_u in subject_vectors.iterrows():

        # compute distance from the vector to all features center
        distances = pairwise_distances(X=np.array(vector_u, ndmin=2), Y=features, metric='euclidean', n_jobs=cpus)
        # distances = distance_matrix(features, np.array(vector_u, ndmin=2))
        # distances = distances.reshape((len(features), ))
        # filter features for which the vector is inside
        distance_close_enough_vec = distances[0] <= max_distance
        features_count = np.sum(distance_close_enough_vec)
        total_feature_count += features_count
        # increment features frequency counter
        if features_count >= 1:
            add_feature_index = np.where(distance_close_enough_vec)
            subject_features_table.loc[subject, feature_list.loc[add_feature_index[0], 'feature_index']] += 1

    print('Finished creating feature table for subject {}, feature count {}, took {}'.format(subject,
                                                                                             total_feature_count,
                                                                                             time.time()-t0))
    return subject_features_table


def test_get_subject_feature_table():
    data = pd.DataFrame()
    data['SUBJECT'] = random.choices(['P1_I1', 'P1_I2', 'P1_I3', 'P1_I4', 'P1_I5', 'P1_I6', 'P1_I7', 'P1_I8'], k=1000)
    for subject in data['SUBJECT'].unique():
        data.loc[data['SUBJECT'] == subject, 'labels'] = random.sample(['healthy', 'celiac'], k=1)[0]
    vectors = pd.DataFrame(np.random.rand(1000, 100))
    feature_list = pd.DataFrame(np.random.rand(100, 100), columns=list(range(100)))
    feature_list['feature_index'] = list(range(100))
    feature_list['max_distance'] = np.random.rand(100, 1) + 4
    cols = feature_list.columns.tolist()
    cols = cols[-2:] + cols[:-2]
    feature_list = feature_list[cols]
    by_subject = data.groupby('SUBJECT')

    result_ids = []
    for subject, frame in by_subject:
        subject_vectors = vectors.iloc[frame.index]
        result_ids += [get_subject_feature_table.remote(subject, feature_list, subject_vectors)]
    features_table = pd.concat([ray.get(id) for id in result_ids])

    # add subject labels column
    features_table['labels'] = None
    for subject, frame in by_subject:
        label = frame['labels'].unique()[0]
        features_table.loc[subject, 'labels'] = label

    return features_table


if __name__ == '__main__':
    main()
