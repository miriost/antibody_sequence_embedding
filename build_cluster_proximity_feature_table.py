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
import gc
from sklearn.metrics.pairwise import pairwise_distances


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('data_file_path', help='the data tsv file path')
    parser.add_argument('vectors_file_path', help='the vectors npy file path')
    parser.add_argument('features_file_path',
                        help='feature list file, contains the list of relevent features, including feature '
                             'center and maximal distance from it')
    parser.add_argument('output_description',  help='description to use inside output file names', type=str)
    parser.add_argument('output_folder', help='Output folder for the feature table')
    parser.add_argument('--dist_metric',
                        help='type of distance to use, default=euclidean', default='euclidean', type=str)
    parser.add_argument('--num_cpus',
                        help='number of cpus to run parallel computing', default=2, type=int)
    parser.add_argument('--same_junction_len', help='Limit cluster to same junction length. Default is False',
                        type=str2bool, default=False)
    parser.add_argument('--same_genes', help='Limit cluster to same v/j_call. Default is False.', type=str2bool,
                        default=False)

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
    same_genes = args.same_genes
    same_junction_len = args.same_junction_len

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

    ray.init(num_cpus=num_cpus, lru_evict=True)

    by_subject = data_file.groupby(id_column)
    # for each subject - add feature frequency columns
    result_ids = []

    features = np.array(features_file['vector'].apply(lambda x: json.loads(x)).to_list())
    max_distance = np.array(features_file.loc[:, 'max_distance']).reshape(1, len(features))

    vectors_file_id = ray.put(vectors_file)
    features_id = ray.put(features)
    max_distance_id = ray.put(max_distance)
    data_file_id = ray.put(data_file)
    features_file_id = ray.put(features_file)

    for subject, frame in by_subject:
        result_ids += [get_subject_feature_table.remote(subject, data_file_id, vectors_file_id, features_id, features_file_id,
                                                        max_distance_id, frame.index, dist_metric, same_genes,
                                                        same_junction_len)]
    features_table = pd.concat([ray.get(res_id) for res_id in result_ids])

    if psutil.virtual_memory().percent >= 80.0:
        gc.collect()

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
def get_subject_feature_table(subject: str, data_file: pd.DataFrame, vectors: np.ndarray, features: np.ndarray,
                              features_file: pd.DataFrame, max_distance: np.ndarray, subject_index, dist_metric,
                              same_genes, same_junction_len):

    features_idx = features_file['feature_index']

    subject_vectors = vectors[subject_index, :]

    print('Start creating feature table for subject {}'.format(subject))
    t0 = time.time()

    by = []
    if same_genes:
        data_file['v_gene'] = data_file['v_call'].str.split('-').apply(lambda x: x[0]).str.split('*').apply(
            lambda x: x[0])
        data_file['j_gene'] = data_file['j_call'].str.split('-').apply(lambda x: x[0]).str.split('*').apply(
            lambda x: x[0])
        by += ['v_gene', 'j_gene']

    if same_junction_len:
        by += ['cdr3_aa_length']

    subject_features_table = pd.DataFrame(0, index=[subject], columns=features_idx.values)

    if len(by) == 0:

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
    else:

        for agg_idx, frame in features_file.groupby(by):
            tmp = data_file
            tmp = tmp[tmp['v_gene'] == agg_idx[0]]
            tmp = tmp[tmp['j_gene'] == agg_idx[1]]
            tmp = tmp[tmp['cdr3_aa_length'] == agg_idx[2]]
            tmp = tmp[tmp['subject.subject_id'] == subject]

            if len(tmp) == 0:
                continue

            frame_features_idx = frame['feature_index']

            if dist_metric == 'manhattan':
                frame_vectors = np.array(tmp['cdr3_aa'].apply(lambda x: [b for b in 'ab'.encode('utf-8')]).to_list())
                frame_features = features[features_idx.to_list(), :]
            else:
                frame_vectors = vectors[tmp.index, :]
                frame_features = features[tmp.index, :]

            # for each vector belonging to the subject
            distances = pairwise_distances(X=frame_vectors, Y=frame_features, metric=dist_metric)
            distance_close_enough_mat = np.logical_or(np.isclose(distances, max_distance, rtol=1e-10, atol=1e-10),
                                                      np.less_equal(distances, max_distance))

            features_count = np.sum(distance_close_enough_mat)
            if features_count >= 1:
                for distance_close_enough_vec in distance_close_enough_mat:
                    if np.sum(distance_close_enough_vec) == 0:
                        continue
                    add_feature_index = np.where(distance_close_enough_vec)
                    subject_features_table.loc[subject, frame_features_idx.iloc[add_feature_index[0]]] += 1

            print('Finished creating feature table for subject {}, feature count {}, took {}'.format(subject,
                                                                                                     features_count,
                                                                                                     time.time() - t0))

    return subject_features_table


if __name__ == '__main__':
    main()

