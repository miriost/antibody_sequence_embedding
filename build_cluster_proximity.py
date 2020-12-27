import os
import argparse
import pandas as pd
import numpy as np
import time
import sys
import math
from sklearn.metrics.pairwise import pairwise_distances
import ray
import psutil
import gc
from sklearn.cluster import AgglomerativeClustering


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('data_file_path', help='the filtered data file path')
    parser.add_argument('vectors_file_path', help='the name of the column with the vector')
    parser.add_argument('output_desc', help='description prefix for the output file names')
    parser.add_argument('output_folder_path', help='Output folder for the output knn files')
    parser.add_argument('--cluster_size', help='size of the cluster, default is 100', type=int, default=100)
    parser.add_argument('--same_junction_len', help='Limit cluster to same junction length. Default is False',
                        type=str2bool, default=False)
    parser.add_argument('--same_genes', help='Limit cluster to same v/j_call. Default is False.', type=str2bool,
                        default=False)
    parser.add_argument('--search_knn', help='Do KNN search - can skip this step if cluster_neighbors column is '
                                             'already available. Default is True.', type=str2bool, default=True)
    parser.add_argument('--analyze_cluster', help='Do cluster label analysis - can skip this step. Default is True.',
                        type=str2bool, default=True)
    parser.add_argument('--dist_metric',
                        help='type of distance to use, default=euclidean', type=str, default='euclidean')
    parser.add_argument('--step', help='How many rows to calculate in parallel, default is 100.',
                        type=int, default=100)
    parser.add_argument('--num_cpus', help='How many cpus are available for ray threads', type=int)
    parser.add_argument('--do_clustering', help='Instead of KNN do full linkage hierarchical clustering', type=bool,
                        default=False)
    parser.add_argument('--distance_th', help='Distance th for the full linkage hierarchical clustering', type=float,
                        default=0.15)
    parser.add_argument('--cluster_id_column', help='the name of the column to store teh cluster_id, default is '
                                                    '"cluster_id"', default='cluster_id')
    parser.add_argument('--thread_memory', help='thread memory size of ray.init()', type=int)

    args = parser.parse_args()

    data_file_path = args.data_file_path
    vectors_file_path = args.vectors_file_path

    if not os.path.isfile(data_file_path):
        print('Data file error, make sure data file path: {}\nExiting...'.format(data_file_path))
        sys.exit(1)

    if not os.path.isfile(vectors_file_path):
        print('Data file error, make sure vectors file path: {}\nExiting...'.format(vectors_file_path))
        sys.exit(1)

    num_cpus = args.num_cpus
    if num_cpus is None:
        num_cpus = psutil.cpu_count()

    thread_memory = args.thread_memory

    if thread_memory is not None and thread_memory > 0:
        ray.init(num_cpus=num_cpus, lru_evict=True, memory=thread_memory, object_store_memory=thread_memory)
    else:
        ray.init(num_cpus=num_cpus, lru_evict=True)
   
    id_column = 'subject.subject_id'
    cluster_size = args.cluster_size
    dist_metric = args.dist_metric
    step = args.step
    same_genes = args.same_genes
    same_junction_len = args.same_junction_len
    output_desc = args.output_desc
    distance_th = args.distance_th
    cluster_id_column = args.cluster_id_column
    output_dir = args.output_folder_path

    data_file = pd.read_csv(args.data_file_path, sep='\t')
    print('loaded data file')
    data_file['index'] = data_file.index

    vectors_file = np.load(vectors_file_path)
    print('loaded vectors file')

    if vectors_file.shape[0] != data_file.shape[0]:
        print('mismatch between data_file and vectors_file length\nExiting...')
        sys.exit(1)

    distances_file_path = os.path.join(output_dir, output_desc + '_distances.npy')
    neighbors_file_path = os.path.join(output_dir, output_desc + '_neighbors.npy')
    subjects_file_path = os.path.join(output_dir, output_desc + '_subjects.npy')

    if args.do_clustering is True:

        if cluster_id_column in data_file.columns:
            print('{} column already exists in data_file, skipping'.format(cluster_id_column))
            return

        data_file = find_clusters(data_file, vectors_file, same_junction_len, same_genes, dist_metric, distance_th,
                                  cluster_id_column)
        data_file.to_csv(data_file_path, sep='\t', index=False)

        return

    if args.search_knn is True:
        data_file = search_knn(data_file, vectors_file, cluster_size, same_junction_len, same_genes,
                               dist_metric, step)

        distances_map = np.array(data_file['cluster_distances'].to_list())
        np.save(distances_file_path, distances_map)
        del distances_map

        neighbors_map = np.array(data_file['cluster_neighbors'].to_list())
        np.save(neighbors_file_path, neighbors_map)
        del neighbors_map

    elif 'cluster_neighbors' not in data_file.columns:

        distances_map = np.load(distances_file_path)
        data_file['cluster_distances'] = pd.Series(distances_map[:, :].tolist())
        del distances_map

        neighbors_map = np.loadtxt(neighbors_file_path, dtype=object)
        data_file['cluster_neighbors'] = pd.Series(neighbors_map[:, :].tolist())
        del neighbors_map

    if args.analyze_cluster is True:
        data_file = analyze_data(data_file, id_column, num_cpus)
        subjects_map = np.array(data_file['cluster_subjects'].to_list())
        np.savetxt(subjects_file_path, subjects_map, fmt='%s')
        del subjects_map


def find_clusters(data_file: pd.DataFrame, vectors_file: np.array, same_junction_len, same_genes, dist_metric,
                  distance_th, cluster_id_column):

    data_file[cluster_id_column] = None
    vectors_id = ray.put(vectors_file)

    by = []
    if same_genes:
        data_file['v_gene'] = data_file['v_call'].str.split('-').apply(lambda x: x[0]).str.split('*').apply(
            lambda x: x[0])
        data_file['j_gene'] = data_file['j_call'].str.split('-').apply(lambda x: x[0]).str.split('*').apply(
            lambda x: x[0])
        by += ['v_gene', 'j_gene']

    if same_junction_len:
        by += ['cdr3_aa_length']

    if len(by) == 0:
        data_file.loc[:, cluster_id_column] = ray.get(do_agglomerative_clustering.remote(vectors_id,
                                                                                         data_file.index.tolist(),
                                                                                         dist_metric,
                                                                                         distance_th))
        print('found {} clusters'.format(len(data_file.loc[cluster_id_column].unique())))
        return data_file

    results_ids = []
    cluster_max_id = 0

    sequences_completed = 0
    for agg_idx, frame in data_file.groupby(by):

        if len(frame) == 1:
            data_file.loc[frame.index, cluster_id_column] = cluster_max_id
            cluster_max_id += 1
            sequences_completed += 1
            continue

        if dist_metric == 'manhattan':
            frame_distance_th = int(distance_th * agg_idx[2])
            frame_vectors = np.array(frame['cdr3_aa'].apply(lambda x: [b for b in 'ab'.encode('utf-8')]).to_list())
            results_ids += [(frame.index, do_agglomerative_clustering.remote(frame_vectors, list(range(frame.shape[0])),
                                                                             dist_metric, frame_distance_th))]
        else:
            results_ids += [(frame.index, do_agglomerative_clustering.remote(vectors_id, frame.index.tolist(),
                                                                             dist_metric, distance_th))]

    for (frame_index, res_id) in results_ids:

        cluster_labels = np.copy(ray.get(res_id))
        cluster_labels += cluster_max_id
        cluster_max_id += len(cluster_labels)
        data_file.loc[frame_index, cluster_id_column] = cluster_labels

        sequences_completed += len(frame_index)

        print('{:.2f}% of sequences completed so far'.format(sequences_completed * 100 / len(data_file)))

    print('found {} clusters'.format(len(data_file[cluster_id_column].unique())))
    return data_file


@ray.remote
def do_agglomerative_clustering(vectors: np.array, frame_index: list, affinity, distance_threshold):
    frame_vectors = vectors[frame_index, :]
    clustering = AgglomerativeClustering(affinity=affinity,
                                         linkage='complete',
                                         distance_threshold=distance_threshold,
                                         n_clusters=None).fit(frame_vectors)
    return clustering.labels_


def search_knn(data_file: pd.DataFrame, vectors_file: np.array, cluster_size, same_junction_len, same_genes,
               dist_metric, step):

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
                                           vectors=vectors_file,
                                           cluster_size=cluster_size,
                                           unassigned=len(data_file),
                                           dist_metric=dist_metric,
                                           step=step)
        cluster_neighbors = pd.Series(knn_map[:, :].tolist())
        cluster_neighbors.index = data_file.index
        data_file.loc[data_file.index, 'cluster_neighbors'] = cluster_neighbors
        
        cluster_distances = pd.Series(distance_map[:, :].tolist())
        cluster_distances.index = data_file.index
        data_file.loc[data_file.index, 'cluster_distances'] = cluster_distances

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
                                           vectors=vectors_file,
                                           cluster_size=cluster_size,
                                           unassigned=len(frame),
                                           dist_metric=dist_metric,
                                           step=step)
        # len(data_file) is the marker for the unassigned neighbors (not enough candidates)
        index_df = pd.DataFrame(frame['index'].to_list() + [len(data_file)])
        cluster_neighbors = np.array(index_df.transpose())[np.arange(1)[:, None], knn_map[:, :]]
        cluster_neighbors = pd.Series(cluster_neighbors[:, :].tolist())
        cluster_neighbors.index = frame.index
        data_file.loc[frame.index, 'cluster_neighbors'] = cluster_neighbors

        sequences_completed += len(frame)
        
        cluster_distances = pd.Series(distance_map[:, :].tolist())
        cluster_distances.index = frame.index
        data_file.loc[frame.index, 'cluster_distances'] = cluster_distances

        print('{:.2f}% of sequences completed so far'.format(sequences_completed*100/len(data_file)))

    return data_file


def auto_garbage_collect(pct=80.0):
    if psutil.virtual_memory().percent >= pct:
        gc.collect()


def build_maps(data: pd.DataFrame, vectors: np.array, cluster_size, unassigned, dist_metric, step):

    if dist_metric == 'seuclidean':
        vectors_std  = np.std(vectors, axis=0)
        vectors_std_id = ray.put(vectors_std)
    else:
        vectors_std_id = None

    if dist_metric == 'euclidean':
        n_jobs = 1
    else:
        n_jobs = psutil.cpu_count()

    distances_map = np.zeros(shape=[vectors.shape[0], cluster_size+1])
    knn_map = np.ones(shape=[vectors.shape[0], cluster_size+1], dtype=int) * unassigned
    step = min(vectors.shape[0], step)
    partitions = math.ceil(vectors.shape[0] / step)
    ranges = [[round(step*i), min(round(step*(i+1)), vectors.shape[0])] for i in range(partitions)]
    vectors_id = ray.put(vectors)

    sequences_completed = 0
    results_ids = [build_distance_and_knn_maps.remote(vectors_id, sub_row_range, k=cluster_size+1,
                                                      dist_metric=dist_metric, vectors_std=vectors_std_id,
                                                      n_jobs=n_jobs) for sub_row_range in ranges]

    t0 = time.time()
    for i, sub_row_range in enumerate(ranges):
        sub_distances_map, sub_knn_map = ray.get(results_ids[i])
        distances_map[sub_row_range[0]:sub_row_range[1], 0:sub_distances_map.shape[1]] = sub_distances_map
        knn_map[sub_row_range[0]:sub_row_range[1], 0:sub_knn_map.shape[1]] = sub_knn_map
        sequences_completed += sub_knn_map.shape[0]

        del sub_distances_map
        del sub_knn_map
        auto_garbage_collect()
        
        print('{:.2f}% of sequences completed after {} seconds'.format(sequences_completed * 100 / len(data), time.time()-t0))

    return distances_map, knn_map


@ray.remote
def build_distance_and_knn_maps(vectors: np.array, sub_row_range, k, dist_metric='euclidean', vectors_std=None, n_jobs=1):

    if dist_metric == 'seuclidean':
        distances_map = pairwise_distances(X=vectors[sub_row_range[0]:sub_row_range[1], :],
                                           Y=vectors, V=vectors_std,
                                           metric=dist_metric, n_jobs=n_jobs)
    else:
        distances_map = pairwise_distances(X=vectors[sub_row_range[0]:sub_row_range[1], :], Y=vectors,
                                           metric=dist_metric, n_jobs=n_jobs)

    knn_map = np.argpartition(distances_map, k-1, axis=1)[:, 0:k]
    knn_map = knn_map[np.arange(knn_map.shape[0])[:, None],
                      np.argsort(distances_map[np.arange(distances_map.shape[0])[:, None], knn_map])]

    return distances_map[np.arange(distances_map.shape[0])[:, None], knn_map], knn_map


def analyze_data(data_file: pd.DataFrame, id_column, num_cpus):

    knn_map = np.array(data_file['cluster_neighbors'].tolist())

    vector_subjects = pd.DataFrame(list(map(lambda x: [x], data_file[id_column])) + [[None]])
    vector_subjects_id = ray.put(vector_subjects)

    step = knn_map.shape[0] / num_cpus
    ranges = [[round(step * i), round(step * (i + 1))] for i in range(num_cpus)]

    results_ids = []
    for sub_range in ranges:
        results_ids += [analyze_sub_data.remote(data_file.iloc[sub_range[0]:sub_range[1], :], vector_subjects_id,
                                                sub_range)]

    analysis_df = pd.concat([ray.get(result_id) for result_id in results_ids], ignore_index=True)
    analysis_df.set_index(data_file.index, inplace=True)
    data_file['cluster_subjects'] = analysis_df['cluster_subjects']

    return data_file


@ray.remote
def analyze_sub_data(data: pd.DataFrame, vector_subjects:pd.Series, sub_range: tuple):

    print("analyze_sub_data: range {}".format(sub_range))

    knn_map = np.array(data['cluster_neighbors'].tolist())

    sub_output_df = pd.DataFrame()

    t0 = time.time()

    cluster_subjects = np.array(vector_subjects.transpose())[np.arange(1)[:, None], knn_map]
    cluster_subjects = pd.Series(cluster_subjects[:, :].tolist())

    sub_output_df['cluster_subjects'] = cluster_subjects

    print("cluster_subjects column added, took {}".format(time.time() - t0))

    return sub_output_df


if __name__ == '__main__':
    main()




