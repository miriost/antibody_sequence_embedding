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
import psutil
import ray
import time
import json
from sklearn.metrics.pairwise import pairwise_distances


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('data_file_path', help='the data tsv file path')
    parser.add_argument('vectors_file_path', help='the vectors npy file path')
    parser.add_argument('features_file_path',
                        help='feature list file, contains the list of relevent features, including feature '
                             'center and maximal distance from it')
    parser.add_argument('output_description',  help='description to use inside output file names', type=str)
    parser.add_argument('--output_folder', default="./",  help='Output folder for the feature table')
    parser.add_argument('--dist_metric',
                        help='type of distance to use, default=euclidean', default='euclidean', type=str)
    parser.add_argument('--num_cpus',
                        help='number of cpus to run parallel computing', default=2, type=int)
    args = parser.parse_args()

    label_column = 'subject.disease_diagnosis'
    id_column = 'subject.subject_id'
    data_file_path = args.data_file_path
    vectors_file_path = args.vectors_file_path
    features_file_path = args.features_file_path
    num_cpus = args.num_cpus
    dist_metric = args.dist_metric
    output_folder = args.output_folder
    output_description = args.output_description

    if not(os.path.isfile(data_file_path)):
        print('feature file error, make sure file path exists\nExiting...')
        sys.exit(1)

    if not (os.path.isfile(vectors_file_path)):
        print('vectors file error, make sure file path exists\nExiting...')
        sys.exit(1)

    if not(os.path.isfile(features_file_path)):
        print('feature list file error, make sure file path exists\nExiting...')
        sys.exit(1)

    data_file = pd.read_csv(data_file_path, sep='\t')

    vectors_file = np.load(vectors_file_path)
    if vectors_file.shape[0] != data_file.shape[0]:
        print('mismatch between data_file and vectors_file length\nExiting...')
        sys.exit(1)
    print('loaded vectors file')

    features_file = pd.read_csv(features_file_path, sep='\t')
    print('loaded features file')

    if label_column not in data_file.columns:
        print(f'label "{label_column}" column name doesnt exist in data file.\nExiting...')
        sys.exit(1)

    if id_column not in data_file.columns:
        print(f'"{id_column}" column name doesnt exist in data file.\nExiting...')
        sys.exit(1)
    
    if 'feature_index' not in features_file.columns:
        print(f'"feature list file error, no "feature index" column. please check.\n exiting...')
        sys.exit(1)

    if 'vector' not in features_file.columns:
        print(f'"feature list file error, no "feature index" column. please check.\n exiting...')
        sys.exit(1)

    if num_cpus is None:
        num_cpus = psutil.cpu_count()

    ray.init(num_cpus=num_cpus)

    by_subject = data_file.groupby(id_column)
    # for each subject - add feature frequency columns
    result_ids = []

    features = np.array(features_file['vector'].apply(lambda x: json.loads(x)).to_list())
    max_distance = np.array(features_file.loc[:, 'max_distance']).reshape(1, len(features))
    features_idx = features_file['feature_index']

    vectors_file_id = ray.put(vectors_file)
    features_id = ray.put(features)
    max_distance_id = ray.put(max_distance)
    features_idx_id = ray.put(features_idx)

    for subject, frame in by_subject:
        result_ids += [get_subject_feature_table.remote(subject, vectors_file_id, features_id, max_distance_id,
                                                        features_idx_id, frame.index, dist_metric)]
    features_table = pd.concat([ray.get(res_id) for res_id in result_ids])

    # add subject labels column
    features_table[label_column] = None
    for subject, frame in by_subject:
        label = frame[label_column].unique()[0]
        features_table.loc[subject, label_column] = label

    # save to file
    features_table.to_csv(os.path.join(output_folder, output_description + '_feature_table.csv'),
                          index_label='SUBJECT')
      
    print('file saved to ', os.path.join(output_folder, output_description + '_feature_table.csv'))


@ray.remote
def get_subject_feature_table(subject: str, vectors: np.ndarray, features: np.ndarray, max_distance: np.ndarray,
                              features_idx: pd.Series, subject_index, dist_metric):
    subject_vectors = vectors[subject_index, :]

    print('Start creating feature table for subject {}'.format(subject))
    t0 = time.time()
    subject_features_table = pd.DataFrame(0, index=[subject], columns=features_idx.values)
    # for each vector belonging to the subject
    distances = pairwise_distances(X=subject_vectors, Y=features, metric=dist_metric)
    distance_close_enough_mat = np.logical_or(np.isclose(distances, max_distance, rtol=1e-10, atol=1e-10),
                                              np.less_equal(distances, max_distance))

    features_count = np.sum(distance_close_enough_mat)
    if features_count >= 1:
        for distance_close_enough_vec in distance_close_enough_mat:
            if np.sum(distance_close_enough_vec) == 0:
                continue
            add_feature_index = np.where(distance_close_enough_vec)
            subject_features_table.loc[subject, features_idx.iloc[add_feature_index[0]]] += 1

    print('Finished creating feature table for subject {}, feature count {}, took {}'.format(subject,
                                                                                             features_count,
                                                                                             time.time()-t0))
    return subject_features_table


if __name__ == '__main__':
    main()

