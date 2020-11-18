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
import random
import ast


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
    parser.add_argument('--max_distance', help='max allowed distance in cluster, default is unlimited', type=float,
                        default=sys.float_info.max)
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
    parser.add_argument('--step', help='How many rows to calculate in parallel, default is 10k.',
                        type=int, default=10000)
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
    label_column = 'subject.disease_diagnosis'
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
    elif 'cluster_neighbors' not in data_file.columns:
        print("Missing vector_column argument\nExisting...")
        exit(1)
    else:
        data_file['cluster_neighbors'] = data_file['cluster_neighbors'].apply(lambda x: ast.literal_eval(x))

    if args.analyze_cluster is True:
        data_file = analyze_data(data_file, id_column, label_column, cpus)

    data_file.to_csv(args.data_file_path, sep='\t', index=False)


def search_knn(data_file, vector_column, cluster_size, same_junction_len, same_genes, dist_metric, cpus, step):

    data_file['cluster_neighbors'] = None
    data_file['cluster_distances'] = None

    if not same_junction_len:
        # look for k closest neighbors regardless of the junction length
        distance_map, knn_map = build_maps(data=data_file,
                                           vector_column=vector_column,
                                           same_genes=same_genes,
                                           cluster_size=cluster_size,
                                           dist_metric=dist_metric,
                                           cpus=cpus,
                                           step=step)
        data_file.loc[:, 'cluster_neighbors'] = knn_map
        data_file.loc[:, 'median_distance'] = np.median(distance_map, axis=1)
        data_file.loc[:, 'max_distance'] = np.max(distance_map, axis=1)
        data_file.loc[:, 'cluster_distances'] = distance_map

        return data_file

    for cdr3_len, frame in data_file.groupby('cdr3_aa_length'):
        # look for k closest neighbors among sequences with the same junction length)
        distance_map, knn_map = build_maps(data=frame,
                                           vector_column=vector_column,
                                           same_genes=same_genes,
                                           cluster_size=cluster_size,
                                           dist_metric=dist_metric,
                                           cpus=cpus,
                                           step=step)
        cluster_neighbors = np.array(frame[['index']].transpose())[np.arange(1)[:, None], knn_map[:, :]]
        cluster_neighbors = pd.Series(cluster_neighbors[:, :].tolist())
        cluster_neighbors.index = frame.index
        data_file.loc[frame.index, 'cluster_neighbors'] = cluster_neighbors

        data_file.loc[frame.index, 'median_distance'] = np.median(distance_map, axis=1)
        data_file.loc[frame.index, 'max_distance'] = np.max(distance_map, axis=1)

        cluster_distances = pd.Series(distance_map[:, :].tolist())
        cluster_distances.index = frame.index
        data_file.loc[frame.index, 'cluster_distances'] = cluster_distances

    return data_file


def build_maps(data, vector_column, same_genes, cluster_size, dist_metric='euclidean', cpus=2, step=15000):

    vectors = np.array(data[vector_column].apply(lambda x: json.loads(x)).tolist(), ndmin=2)

    if same_genes:
        # add 2 fictitious dimensions, to make sure that junctions with different genes will not be selected to the
        # same cluster as their distance will be too far
        v_gene = data['v_call'].str.split('-').apply(lambda x: x[0])
        j_gene = data['j_call'].str.split('-').apply(lambda x: x[0])
        vectors = np.c_[vectors, np.array(pd.factorize(v_gene)[0] * 10000, ndmin=2).transpose()]
        vectors = np.c_[vectors, np.array(pd.factorize(j_gene)[0] * 1000, ndmin=2).transpose()]

    distances_map = np.zeros(shape=[vectors.shape[0], cluster_size+1])
    knn_map = np.zeros(shape=[vectors.shape[0], cluster_size+1], dtype=int)
    step = max(vectors.shape[0], step)
    partitions = math.ceil(vectors.shape[0] / step)
    ranges = [[round(step*i), min(round(step*(i+1)), vectors.shape[0])] for i in range(partitions)]

    for major_row_range in ranges:
        t0 = time.time()
        print("calling build_sub_map for range {}".format(major_row_range))
        sub_distances_map, sub_knn_map = build_sub_map(vectors, major_row_range, cluster_size=cluster_size,
                                                       dist_metric=dist_metric, cpus=cpus)
        print("build_sub_map: creating sub map (range {}) took {}".format(major_row_range, time.time()-t0))
        distances_map[major_row_range[0]:major_row_range[1], :] = sub_distances_map
        knn_map[major_row_range[0]:major_row_range[1], :] = sub_knn_map

    return distances_map, knn_map


def build_sub_map(vectors, major_row_range, cluster_size, dist_metric='euclidean', cpus=2):
    step = (major_row_range[1]-major_row_range[0]) / cpus
    ranges = [[round(step * i), round(step * (i + 1))] for i in range(cpus)]
    results_ids = []

    knn_map = np.zeros(shape=[major_row_range[1]-major_row_range[0], cluster_size+1], dtype=int)
    distances_map = np.zeros(shape=[major_row_range[1]-major_row_range[0], cluster_size+1])
    for minor_row_range in ranges:
        sub_row_range = [major_row_range[0] + minor_row_range[0], major_row_range[0] + minor_row_range[1]]
        results_ids += [build_distance_and_knn_maps.remote(vectors, sub_row_range, k=cluster_size + 1,
                                                           dist_metric=dist_metric, cpus=cpus)]
    for i, minor_row_range in enumerate(ranges):
        sub_distances_map, sub_knn_map = ray.get(results_ids[i])
        distances_map[minor_row_range[0]:minor_row_range[1], :] = sub_distances_map
        knn_map[minor_row_range[0]:minor_row_range[1], :] = sub_knn_map

    return distances_map, knn_map


@ray.remote
def build_distance_and_knn_maps(vectors, sub_row_range, k, dist_metric='euclidean', cpus=2):
    t0 = time.time()
    print("building distance map for range {}".format(sub_row_range))

    print(vectors.shape)

    distances_map = pairwise_distances(X=vectors[sub_row_range[0]:sub_row_range[1]], Y=vectors, metric=dist_metric,
                                       n_jobs=cpus)
    print("building distance map for range {} took {}".format(sub_row_range, time.time() - t0))

    t0 = time.time()
    print("building knn map for range {}".format(sub_row_range))
    knn_map = np.argpartition(distances_map, k, axis=1)[:, 0:k]
    knn_map = knn_map[np.arange(knn_map.shape[0])[:, None],
                      np.argsort(distances_map[np.arange(distances_map.shape[0])[:, None], knn_map])]
    print("building knn map for range {} took {}".format(sub_row_range, time.time() - t0))

    return distances_map[np.arange(distances_map.shape[0])[:, None], knn_map], knn_map


def analyze_data(data_file: pd.DataFrame, id_column, label_column, cpus):

    data_file['vector_subjects'] = list(map(lambda x: [x], data_file[id_column]))
    knn_map = np.array(data_file['cluster_neighbors'].tolist())
    labels = data_file[label_column].unique().tolist()
    subjects = data_file.loc[:, ].groupby(by=[id_column])[label_column].apply(lambda x: x.iloc[0])

    step = len(knn_map) / cpus
    ranges = [[round(step * i), round(step * (i + 1))] for i in range(cpus)]

    results_ids = []
    for sub_range in ranges:
        results_ids += [analyze_sub_data.remote(data_file, knn_map, sub_range, labels, subjects)]

    analysis_df = pd.concat([ray.get(result_id) for result_id in results_ids], ignore_index=True)

    data_file.loc[:, analysis_df.columns] = analysis_df

    data_file.drop(['vector_subjects'], axis=1, inplace=True)

    return data_file


@ray.remote
def analyze_sub_data(data_file: pd.DataFrame, knn_map: np.ndarray, sub_range: tuple, labels: list,
                     subjects: pd.DataFrame):

    print("analyze_sub_data: range {}".format(sub_range))

    sub_output_df = pd.DataFrame()

    t0 = time.time()
    neighbors = np.array(data_file[['vector_subjects']].transpose())[np.arange(1)[:, None],
                                                                     knn_map[sub_range[0]:sub_range[1]]]
    neighbors = np.array(list(map(lambda x: np.unique(np.concatenate(x)), neighbors)))
    for status in labels:
        sub_output_df[status + '_subjects'] = list(
            map(lambda x: subjects[(subjects.index.isin(x)) & (subjects == status)].index.tolist(), neighbors))

    sub_output_df['how_many_subjects'] = list(map(lambda x: len(x), neighbors))
    print("subjects columns added, took {}".format(time.time() - t0))

    t0 = time.time()
    tmp = pd.concat(list(map(lambda x: subjects[x].value_counts(normalize=True), neighbors)), sort=True, axis=1)
    tmp = tmp.transpose().reset_index(drop=True)
    tmp = tmp.fillna(0)
    sub_output_df[labels] = tmp[labels]
    print("labels columns added, took {}".format(time.time() - t0))

    return sub_output_df


def test_build_dist_and_knn_maps(dim_one=1000, dim_two=10, cluster_size=5, cpus=2):
    data = np.random.rand(dim_one, dim_two)
    t0 = time.time()
    distance_map, knn_map = build_maps(data, cluster_size, dist_metric='euclidean', cpus=cpus, step=10)
    print("building maps completed after {}".format(time.time() - t0))


def test_analyze_sub_data():
    data = pd.DataFrame()
    data['subject'] = random.choices(['P1_I1', 'P1_I2', 'P1_I3', 'P1_I4', 'P1_I5', 'P1_I6', 'P1_I7', 'P1_I8'], k=1000)
    data['labels'] = random.choices(['healthy', 'celiac'], k=1000)
    knn_map = np.array([random.sample(range(1000), 10) for i in range(1000)])
    status_types = data['labels'].unique()

    cpus = 3
    step = len(data) / cpus
    ranges = [[round(step * i), round(step * (i + 1))] for i in range(cpus)]

    results_ids = []
    for sub_range in ranges:
        results_ids += [analyze_sub_data.remote(knn_map, data, sub_range, status_types)]

    output_df = pd.concat([ray.get(result_id) for result_id in results_ids], ignore_index=True)

    return output_df


if __name__ == '__main__':
    main()




