import os
import argparse
import pandas as pd
import numpy as np
import time
import sys
import math
from sklearn.metrics.pairwise import pairwise_distances
import ray
import json
import ast
import random
import string

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', "True"):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', "False"):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('data_file_path', help='the filtered data file path')
    parser.add_argument('vector_column', help='the name of the column with the vector')
    parser.add_argument('--cluster_size', help='size of the cluster, default is 100', type=int, default=100)
    parser.add_argument('--same_junction_len', help='Limit cluster to same junction length. Default is False',
                        type=str2bool, default=False)
    parser.add_argument('--same_genes', help='Limit cluster to same v/j_call. Default is True.', type=str2bool,
                        default=True)
    parser.add_argument('--search_knn', help='Do KNN search - can skip this step if cluster_neighbors column is '
                                             'already available. Default is True.', type=str2bool, default=True)
    parser.add_argument('--analyze_cluster', help='Do cluster label analysis - can skip this step. Default is True.',
                        type=str2bool, default=True)
    parser.add_argument('--dist_metric',
                        help='type of distance to use, default=euclidean', type=str, default='euclidean')
    parser.add_argument('--cpus', help='How many cores to run in parallel -  default is 1.', type=int, default=1)
    parser.add_argument('--step', help='How many rows to calculate in parallel, default is 15k.',
                        type=int, default=15000)
    parser.add_argument('--thread_memory', help='memory size for ray thread (bytes)', type=int)

    args = parser.parse_args()

    if not os.path.isfile(args.data_file_path):
        print('Data file error, make sure data file path: {}\nExiting...'.format(args.data_file_path))
        sys.exit(1)

    if args.vector_column is None:
        print("Missing vector_column argument\nExisting...")
        sys.exit(1)

    if args.thread_memory is not None:
        ray.init(memory=args.thread_memory, object_store_memory=args.thread_memory)
    else:
        ray.init()
   
    vector_column = args.vector_column
    id_column = 'subject.subject_id'
    cluster_size = args.cluster_size
    dist_metric = args.dist_metric
    cpus = args.cpus
    step = args.step
    same_genes = args.same_genes
    same_junction_len = args.same_junction_len

    data_file = pd.read_csv(args.data_file_path, sep='\t')
    data_file['index'] = data_file.index

    if args.search_knn is True:
        data_file = search_knn(data_file, vector_column, cluster_size, same_junction_len, same_genes,
                               dist_metric, cpus, step)
        data_file.to_csv(args.data_file_path, sep='\t', index=False)
    elif 'cluster_neighbors' not in data_file.columns:
        print("Missing cluster_neighbors column\nExisting...")
        exit(1)
    else:
        data_file['cluster_neighbors'] = data_file['cluster_neighbors'].apply(lambda x: ast.literal_eval(x))

    if args.analyze_cluster is True:
        data_file = analyze_data(data_file, id_column, cpus)
        data_file.to_csv(args.data_file_path, sep='\t', index=False)


def search_knn(data_file, vector_column, cluster_size, same_junction_len, same_genes, dist_metric, cpus, step):

    data_file['cluster_neighbors'] = None
    data_file['cluster_distances'] = None

    by = []
    if same_genes:
        data_file['v_gene'] = data_file['v_call'].str.split('-').apply(lambda x: x[0]).str.split('*').apply(lambda x: x[0])
        data_file['j_gene'] = data_file['j_call'].str.split('-').apply(lambda x: x[0]).str.split('*').apply(lambda x: x[0])
        by += ['v_gene', 'j_gene']

    if same_junction_len:
        by += ['cdr3_aa_length']

    if len(by) == 0:
        # look for k closest neighbors regardless of the junction length
        distance_map, knn_map = build_maps(data=data_file,
                                           vector_column=vector_column,
                                           same_genes=same_genes,
                                           cluster_size=cluster_size,
                                           dist_metric=dist_metric,
                                           cpus=cpus,
                                           step=step)
        data_file.loc[:, 'cluster_neighbors'] = knn_map
        data_file.loc[:, 'cluster_distances'] = distance_map

        return data_file

    sequences_completed = 0
    for agg_idx, frame in data_file.groupby(by):

        if len(frame) == 1:
            # handle the edge case of a frame of size one
            tmp = pd.Series([np.zeros(cluster_size + 1).tolist()], index=frame.index)
            data_file.loc[frame.index, 'cluster_distances'] = tmp
            tmp = pd.Series([np.zeros(cluster_size + 1, dtype=int).tolist()], index=frame.index)
            tmp.iloc[0][0] = frame.iloc[0]['index']
            data_file.loc[frame.index, 'cluster_neighbors'] = tmp
            sequences_completed += len(frame)
            continue

        # look for k closest neighbors among sequences with the same junction length)
        distance_map, knn_map = build_maps(data=frame,
                                           vector_column=vector_column,
                                           cluster_size=cluster_size,
                                           unassigned=len(frame),
                                           dist_metric=dist_metric,
                                           cpus=cpus,
                                           step=step)
        # len(data_file) is the marker for the unassigned neighbors (not enough candidates)
        index_df = pd.DataFrame(frame['index'].to_list() + [len(data_file)])
        cluster_neighbors = np.array(index_df.transpose())[np.arange(1)[:, None], knn_map[:, :]]
        cluster_neighbors = pd.Series(cluster_neighbors[:, :].tolist())
        cluster_neighbors.index = frame.index
        data_file.loc[frame.index, 'cluster_neighbors'] = cluster_neighbors

        data_file.loc[frame.index, 'median_distance'] = np.median(distance_map, axis=1)
        data_file.loc[frame.index, 'max_distance'] = np.max(distance_map, axis=1)

        cluster_distances = pd.Series(distance_map[:, :].tolist())
        cluster_distances.index = frame.index
        data_file.loc[frame.index, 'cluster_distances'] = cluster_distances

        sequences_completed += len(frame)
        print('{:.2f}% of sequences completed so far'.format(sequences_completed*100/len(data_file)))

    return data_file


def build_maps(data, vector_column, cluster_size, unassigned, dist_metric, cpus, step):

    vectors = np.array(data[vector_column].apply(lambda x: json.loads(x)).tolist(), ndmin=2)

    distances_map = np.zeros(shape=[vectors.shape[0], cluster_size+1])
    knn_map = np.ones(shape=[vectors.shape[0], cluster_size+1], dtype=int) * unassigned
    step = min(vectors.shape[0], step)
    partitions = math.ceil(vectors.shape[0] / step)
    ranges = [[round(step*i), min(round(step*(i+1)), vectors.shape[0])] for i in range(partitions)]

    for major_row_range in ranges:
        sub_distances_map, sub_knn_map = build_sub_map(vectors,
                                                       major_row_range,
                                                       cluster_size=min(cluster_size+1, len(data)),
                                                       unassigned=unassigned,
                                                       dist_metric=dist_metric,
                                                       cpus=cpus)
        distances_map[major_row_range[0]:major_row_range[1], 0:sub_distances_map.shape[1]] = sub_distances_map
        knn_map[major_row_range[0]:major_row_range[1], 0:sub_knn_map.shape[1]] = sub_knn_map

    return distances_map, knn_map


def build_sub_map(vectors, major_row_range, cluster_size, unassigned, dist_metric='euclidean', cpus=2):
    if major_row_range[1]-major_row_range[0] < cpus * 3:
        ranges = [[0, major_row_range[1]-major_row_range[0]]]
    else:
        step = (major_row_range[1]-major_row_range[0]) / cpus
        ranges = [[round(step * i), round(step * (i + 1))] for i in range(cpus)]
    results_ids = []

    knn_map = np.ones(shape=[major_row_range[1]-major_row_range[0], cluster_size], dtype=int) * unassigned
    distances_map = np.zeros(shape=[major_row_range[1]-major_row_range[0], cluster_size])
    for minor_row_range in ranges:
        sub_row_range = [major_row_range[0] + minor_row_range[0], major_row_range[0] + minor_row_range[1]]
        results_ids += [build_distance_and_knn_maps.remote(vectors, sub_row_range, k=cluster_size,
                                                           dist_metric=dist_metric, cpus=cpus)]
    for i, minor_row_range in enumerate(ranges):
        sub_distances_map, sub_knn_map = ray.get(results_ids[i])
        distances_map[minor_row_range[0]:minor_row_range[1], 0:sub_distances_map.shape[1]] = sub_distances_map
        knn_map[minor_row_range[0]:minor_row_range[1], 0:sub_knn_map.shape[1]] = sub_knn_map

    return distances_map, knn_map


@ray.remote
def build_distance_and_knn_maps(vectors, sub_row_range, k, dist_metric='euclidean', cpus=2):

    distances_map = pairwise_distances(X=vectors[sub_row_range[0]:sub_row_range[1], :], Y=vectors, metric=dist_metric,
                                       n_jobs=cpus)

    knn_map = np.argpartition(distances_map, k-1, axis=1)[:, 0:k]
    knn_map = knn_map[np.arange(knn_map.shape[0])[:, None],
                      np.argsort(distances_map[np.arange(distances_map.shape[0])[:, None], knn_map])]

    return distances_map[np.arange(distances_map.shape[0])[:, None], knn_map], knn_map


def analyze_data(data_file: pd.DataFrame, id_column, cpus):

    knn_map = np.array(data_file['cluster_neighbors'].tolist())

    vector_subjects = pd.DataFrame(list(map(lambda x: [x], data_file[id_column])) + [[None]])

    step = knn_map.shape[0] / cpus
    ranges = [[round(step * i), round(step * (i + 1))] for i in range(cpus)]

    results_ids = []
    for sub_range in ranges:
        results_ids += [analyze_sub_data.remote(vector_subjects, knn_map, sub_range)]

    analysis_df = pd.concat([ray.get(result_id) for result_id in results_ids], ignore_index=True)
    analysis_df.set_index(data_file.index, inplace=True)
    data_file.loc['cluster_subjects'] = analysis_df['cluster_subjects']

    return data_file


@ray.remote
def analyze_sub_data(vector_subjects: pd.Series, knn_map: np.ndarray, sub_range: tuple):

    print("analyze_sub_data: range {}".format(sub_range))

    sub_output_df = pd.DataFrame(columns=['cluster_subjects'], index=range(sub_range[0], sub_range[1]))

    t0 = time.time()

    cluster_subjects = np.array(vector_subjects.transpose())[np.arange(1)[:, None], knn_map[sub_range[0]:sub_range[1]]]
    cluster_subjects = pd.Series(cluster_subjects[:, :].tolist(), index=range(sub_range[0], sub_range[1]))

    sub_output_df.loc[sub_output_df.index, 'cluster_subjects'] = cluster_subjects

    print("cluster_subjects column added, took {}".format(time.time() - t0))

    return sub_output_df


if __name__ == '__main__':
    main()




