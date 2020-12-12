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
import Levenshtein
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file_path', help='tsv data file path')
    parser.add_argument('vectors_file_path', help='npy vectors file')
    parser.add_argument('--num_sequences', help='add num_sequences to matrix (row dim), default is 10k', type=int,
                        default=10000)
    parser.add_argument('--knn', help='add k nearest neighbors neighbors to matrix (columns dim), default is 400',
                        type=int, default=400)
    parser.add_argument('--distance_metrics', help="semicolon separated list of distance metrics for comparision, "
                                                   "options are: 'braycurtis', 'canberra', 'chebyshev', 'cityblock', "
                                                   "'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard',"
                                                   " 'jensenshannon', 'kulsinski', 'mahalanobis', 'matching', "
                                                   "'minkowski<p-norm>', 'rogerstanimoto', 'russellrao', 'seuclidean', "
                                                   "'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'",
                        type=str)
    parser.add_argument('--same_cdr3_length', help='search knn among sequences with the same cdr3 length, default is '
                                                   'False', default=False, type=str2bool)
    parser.add_argument('--plot_dim_reduction', help='plot selected clusters after dimension reduction, default is '
                                                     'False', default=False, type=str2bool)
    parser.add_argument('--random_seed', help='Random seed for sampling the sequences, default is 0.', type=int,
                        default=0)
    parser.add_argument('--step', help='How many rows to calculate in parallel, default is 200.', type=int, default=200)
    parser.add_argument('--num_cpus', help='How many cpus are available for ray threads', type=int)

    args = parser.parse_args()

    execute(args)


def auto_garbage_collect(pct=80.0):
    if psutil.virtual_memory().percent >= pct:
        gc.collect()


def execute(args):

    data_file_path = args.data_file_path
    vectors_file_path = args.vectors_file_path
    num_cpus = args.num_cpus

    if not os.path.isfile(data_file_path) or data_file_path[:-4] == '.tsv':
        print('Data file error - file not exists or suffix is not .tsv, make sure data file path: {}\n'
              'Exiting...'.format(data_file_path))
        sys.exit(1)

    if not os.path.isfile(vectors_file_path) or vectors_file_path[:-4] == '.npy':
        print('Vectors file error - file not exists or suffix is not .npy, make sure vectors file path: {}\n'
              'Exiting...'.format(vectors_file_path))
        sys.exit(1)

    data_file = pd.read_csv(args.data_file_path, sep='\t')
    vectors = np.load(vectors_file_path)

    if vectors.shape[0] != data_file.shape[0]:
        print('mismatch between data_file and vectors_file length\nExiting...')
        sys.exit(1)

    if num_cpus is None:
        num_cpus = psutil.cpu_count()

    ray.init(num_cpus=num_cpus)

    knn = args.knn
    random_seed = args.random_seed
    num_sequences = args.num_sequences
    if args.distance_metrics is None:
        distance_metrics = []
    else:
        distance_metrics = args.distance_metrics.split(';')
    step = args.step
    same_cdr3_length = args.same_cdr3_length
    plot_dim_reduction = args.plot_dim_reduction

    # sample the sequences
    if same_cdr3_length:
        prefix = 'same_cdr3_length_'
    else:
        prefix = ''

    lev_dist_map_file = prefix + 'levenshtein_distances_map.npy'
    lev_knn_map_file = prefix + 'knn_map.npy'

    np.random.seed(random_seed)
    samples = np.sort(np.random.choice(data_file.index, replace=False, size=num_sequences)).tolist()

    if os.path.isfile(lev_dist_map_file) and os.path.isfile(lev_knn_map_file):
        print('loading knn map from file {}'.format(lev_knn_map_file))
        knn_map = np.load(lev_knn_map_file, dtype=int)

    else:
        X_sequences = data_file.loc[samples, 'cdr3_aa']
        Y_sequences = data_file['cdr3_aa']

        distances_map, knn_map = create_knn_map(X_sequences, Y_sequences, knn, step, same_cdr3_length)
        np.save(lev_dist_map_file, distances_map)
        data_file.loc[len(Y_sequences), 'document._id'] = None
        tagged_knn_map = np.array(np.apply_along_axis(lambda x: data_file.loc[x, 'document._id'].to_list(),
                                                      1, knn_map).tolist())
        np.save(prefix + 'knn_map.npy', tagged_knn_map)
        np.savetxt(prefix + 'tagged_knn_map.npy', tagged_knn_map, fmt='%s')
        data_file.drop([len(Y_sequences)], axis=0, inplace=True)

    X_vectors = vectors[samples, :]

    for dist_metric in distance_metrics:
        distances_map = create_dist_map(X_vectors, vectors, knn_map, dist_metric, step)
        np.save(prefix + dist_metric + '_distances_map.npy', distances_map)

    if plot_dim_reduction is False:
        return

    indexing = data_file.index[data_file['document._id'].isin(np.unique(tagged_knn_map))]
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(vectors)

    plt.clf()
    fig, axes = plt.subplots(ncols=2, nrows=2, constrained_layout=True)
    np.random.seed(random_seed)
    for i in range(4):
        plot_samples = np.sort(np.random.choice(tagged_knn_map.shape[0], replace=False, size=10))
        ax = axes[int(i / axes.shape[1]), i % axes.shape[0]]
        ax.set_xticks([])
        ax.set_yticks([])
        for sample_index in plot_samples:
            samples_indexing = data_file.index[data_file['document._id'].isin(np.unique(tagged_knn_map[sample_index, :]))]
            reduced_vectors_to_plot = reduced_vectors[indexing.isin(samples_indexing), :]
            sns.scatterplot(x=reduced_vectors_to_plot[:, 0], y=reduced_vectors_to_plot[:, 1], ax=ax)

    print('saving file dim_reduction.png')
    fig.savefig('dim_reduction.png')


def create_knn_map(X_sequences, Y_sequences, knn, step, same_cdr3_length):

    distances_map = np.zeros(shape=[X_sequences.shape[0], knn + 1], dtype=float)
    knn_map = np.ones(shape=[X_sequences.shape[0], knn + 1], dtype=int)
    sequences_completed = 0

    if same_cdr3_length is False:

        X_sequences_id = ray.put(X_sequences)
        Y_sequences_id = ray.put(Y_sequences)

        partitions = math.ceil(X_sequences.shape[0] / step)
        ranges = [[round(step * i), min(round(step * (i + 1)), X_sequences.shape[0])] for i in range(partitions)]
        results_ids = [build_knn_map.remote(X_sequences_id, Y_sequences_id, sub_row_range, k=knn+1)
                       for sub_row_range in ranges]

        t0 = time.time()
        for i, sub_row_range in enumerate(ranges):
            sub_distances_map, sub_knn_map = ray.get(results_ids[i])
            distances_map[sub_row_range[0]:sub_row_range[1], 0:sub_distances_map.shape[1]] = sub_distances_map
            knn_map[sub_row_range[0]:sub_row_range[1], 0:sub_knn_map.shape[1]] = sub_knn_map
            sequences_completed += sub_knn_map.shape[0]

            del sub_distances_map
            del sub_knn_map
            auto_garbage_collect()

            print('dist_metric {:.2f}% of sequences completed after {} seconds'.format(sequences_completed * 100 / len(X_sequences),
                                                                                       time.time() - t0))
        # normalize to values between [-1, 1]
        distances_map /= np.nanmax(np.abs(distances_map))
        return distances_map, knn_map

    distances_map[:] = np.nan
    knn_map[:] = len(Y_sequences)
    X_sequences = X_sequences.reset_index(drop=True).copy(deep=True)
    cdr3_aa_lengths = X_sequences.str.len()
    t0 = time.time()
    for cdr3_aa_len in cdr3_aa_lengths.unique():

        X_frame = X_sequences[cdr3_aa_lengths == cdr3_aa_len].copy(deep=True)
        Y_frame = Y_sequences[Y_sequences.str.len() == cdr3_aa_len]
        if len(X_frame) == 1:
            # handle the edge case of a frame of size one
            distances_map[X_frame.index, 0] = 0
            knn_map[X_frame.index, 0] = X_frame.index
            sequences_completed += 1
            continue

        X_frame_id = ray.put(X_frame)
        Y_frame_id = ray.put(Y_frame)

        frame_step = min(step, len(X_frame))
        partitions = math.ceil(X_frame.shape[0] / frame_step)
        ranges = [[round(step * i), min(round(frame_step * (i + 1)), X_frame.shape[0])] for i in range(partitions)]
        results_ids = [build_knn_map.remote(X_frame_id, Y_frame_id, sub_row_range, k=min(knn+1, len(Y_frame)))
                       for sub_row_range in ranges]

        index_df = pd.DataFrame(Y_frame.index.tolist())
        for i, sub_row_range in enumerate(ranges):
            sub_distances_map, sub_knn_map = ray.get(results_ids[i])
            rows_index = X_frame.index[sub_row_range[0]:sub_row_range[1]]
            distances_map[rows_index, 0:sub_distances_map.shape[1]] = sub_distances_map

            translated_sub_knn_map = np.array(index_df.transpose())[np.arange(1)[:, None], sub_knn_map[:, :]]
            knn_map[rows_index, 0:sub_knn_map.shape[1]] = translated_sub_knn_map

            sequences_completed += sub_knn_map.shape[0]

            del sub_distances_map
            del sub_knn_map
            auto_garbage_collect()

            print('dist_metric {:.2f}% of sequences completed after {} seconds'.format(sequences_completed * 100 / len(X_sequences),
                                                                                       time.time() - t0))

    # normalize to values between [-1, 1]
    distances_map /= np.nanmax(np.abs(distances_map))
    return distances_map, knn_map


@ray.remote
def build_knn_map(X_sequences: pd.Series, Y_sequences: pd.Series, sub_row_range, k):

    sub_X = X_sequences.iloc[sub_row_range[0]:sub_row_range[1]]
    distances_map = np.array(sub_X.apply(lambda x: Y_sequences.apply(lambda y: Levenshtein.distance(x, y)).tolist()).tolist())

    knn_map = np.argpartition(distances_map, k - 1, axis=1)[:, 0:k]
    knn_map = knn_map[np.arange(knn_map.shape[0])[:, None],
                      np.argsort(distances_map[np.arange(distances_map.shape[0])[:, None], knn_map])]

    distances_map = distances_map[np.arange(distances_map.shape[0])[:, None], knn_map]

    return distances_map, knn_map


def create_dist_map(X_vectors, Y_vectors, knn_map, dist_metric, step):

    X_vectors_id = ray.put(X_vectors)
    Y_vectors_id = ray.put(Y_vectors)

    if dist_metric == 'seuclidean':
        vectors_std = np.std(X_vectors, axis=0)
        vectors_std_id = ray.put(vectors_std)
    else:
        vectors_std_id = None

    if dist_metric == 'euclidean':
        n_jobs = 1
    else:
        n_jobs = psutil.cpu_count()

    knn_map_id = ray.put(knn_map)
    distances_map = np.zeros(shape=[X_vectors.shape[0], knn_map.shape[1]], dtype=float)
    partitions = math.ceil(X_vectors.shape[0] / step)
    ranges = [[round(step * i), min(round(step * (i + 1)), X_vectors.shape[0])] for i in range(partitions)]

    sequences_completed = 0
    results_ids = [build_dist_map.remote(X_vectors_id, Y_vectors_id, sub_row_range, knn_map_id, dist_metric, n_jobs,
                                         vectors_std=vectors_std_id) for sub_row_range in ranges]

    t0 = time.time()
    for i, sub_row_range in enumerate(ranges):
        sub_distances_map = ray.get(results_ids[i])
        distances_map[sub_row_range[0]:sub_row_range[1], 0:sub_distances_map.shape[1]] = sub_distances_map
        sequences_completed += sub_distances_map.shape[0]

        del sub_distances_map
        auto_garbage_collect()

        print(dist_metric + ' {:.2f}% of sequences completed after {} seconds'.format(sequences_completed * 100 / len(X_vectors),
                                                                                      time.time() - t0))
    # normalize to values between [-1, 1]
    distances_map /= np.nanmax(np.abs(distances_map))
    return distances_map


@ray.remote
def build_dist_map(X_vectors: np.array, Y_vectors: np.array, sub_row_range, knn_map, dist_metric, n_jobs,
                   vectors_std=None):

    def build_row(idx):
        seq_vector = X_vectors[idx, :].reshape(1, -1)
        knn_vectors = Y_vectors[[i for i in knn_map[idx, :] if i < Y_vectors.shape[0]], :]
        row = np.zeros(knn_map.shape[1])
        row[:] = np.nan
        if dist_metric == 'seuclidean':
            row[0:len(knn_vectors)] = pairwise_distances(X=seq_vector, Y=knn_vectors, V=vectors_std, metric=dist_metric,
                                                         n_jobs=n_jobs).tolist()[0]
        elif dist_metric.startswith('minkowski'):
            p = dist_metric.split('minkowski')[1]
            try:
                p_norm = int(p)
            except ValueError:
                print('invalid p suffix to minkowski distance ({}), defaulting to p=1'.format(dist_metric))
                p_norm = 1
            row[0:len(knn_vectors)] = pairwise_distances(X=seq_vector, Y=knn_vectors, p=p_norm, metric='minkowski',
                                                         n_jobs=n_jobs).tolist()[0]
        else:
            row[0:len(knn_vectors)] = pairwise_distances(X=seq_vector, Y=knn_vectors, metric=dist_metric,
                                                         n_jobs=n_jobs).tolist()[0]

        return row.tolist()

    distances_map = np.array(list(map(build_row, range(sub_row_range[0], sub_row_range[1]))))

    return distances_map


if __name__ == '__main__':
    main()




