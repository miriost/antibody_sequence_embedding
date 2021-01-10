import os
import argparse
import pandas as pd
import numpy as np
import time
import sys
import ray
import psutil
import gc
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import mode


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('data_file_path', help='the filtered data file path')
    parser.add_argument('output_desc', help='description prefix for the output file names')
    parser.add_argument('output_folder_path', help='Output folder for the output knn files')
    parser.add_argument('subject_col', help='name of the column with the subjects ids')
    parser.add_argument('label_col', help='name of the column with the disease diagnosis')

    parser.add_argument('--similarity_th', help='similarity percentage cutoff value as percentage of the clustering. '
                                                'Default is 0.85', type=int, default=0.85)
    parser.add_argument('--subjects_th', help='th filter by min subjects, default is 5', default=5, type=int)
    parser.add_argument('--significance_th', help='th filter by min significance, default is 0.75', default=0.75,
                        type=float)
    parser.add_argument('--cluster_id_column', help='the name of the column to store the cluster_id, default is '
                                                    '"cluster_id"', default='cluster_id')
    parser.add_argument('--num_cpus', help='How many cpus are available for ray threads', type=int)
    parser.add_argument('--thread_memory', help='thread memory size of ray.init()', type=int)

    args = parser.parse_args()

    data_file_path = args.data_file_path
    if not os.path.isfile(data_file_path):
        print('Data file error, make sure data file path: {}\nExiting...'.format(data_file_path))
        sys.exit(1)

    num_cpus = args.num_cpus
    if num_cpus is None:
        num_cpus = psutil.cpu_count()

    thread_memory = args.thread_memory

    if thread_memory is not None and thread_memory > 0:
        ray.init(num_cpus=num_cpus, lru_evict=True, memory=thread_memory, object_store_memory=thread_memory)
    else:
        ray.init(num_cpus=num_cpus, lru_evict=True)
   
    output_desc = args.output_desc
    output_folder_path = args.output_folder_path
    similarity_th = args.similarity_th
    cluster_id_column = args.cluster_id_column
    subjects_th = args.subjects_th
    significance_th = args.significance_th
    subject_col = args.subject_col
    label_col = args.label_col

    data_file = pd.read_csv(args.data_file_path, sep='\t')
    print('loaded data file')

    vectors = np.array(data_file['cdr3_aa'].apply(lambda x: [b for b in 'ab'.encode('utf-8')]).to_list())

    if cluster_id_column in data_file.columns:
        print('{} column already exists in data_file, skipping'.format(cluster_id_column))
        return

    data_file = find_clusters(data_file, vectors, cluster_id_column, similarity_th)
    data_file.to_csv(data_file_path, sep='\t', index=False)

    selected_clusters = filter_clusters(data_file, vectors,
                                        cluster_id_column, subject_col, label_col,
                                        similarity_th, subjects_th, significance_th)

    selected_clusters.to_csv(os.path.join(output_folder_path, output_desc + '.tsv'), sep='\t',
                             index_label='feature_index')
    print('feature list is saved to {}'.format(os.path.join(output_folder_path, output_desc + '.tsv')))


def filter_clusters(data_file: pd.DataFrame, vectors: np.array, cluster_id_column, subject_col, label_col,
                    similarity_th, subjects_th, significance_th):

    selected_clusters = pd.DataFrame(columns=['feature_index', 'vector', 'max_distance', 'v_gene', 'j_gene',
                                              'cdr3_aa_length'])

    t0 = time.time()
    for cluster_id, frame in data_file.groupby([cluster_id_column]):

        if len(frame[subject_col].unique()) < subjects_th:
            continue

        frame_value_counts = frame[label_col].value_counts(normalize=True)
        if frame_value_counts.max() < significance_th:
            continue

        frame_vectors = vectors[frame.index, :]
        cluster_center = mode(frame_vectors)[0].tolist()
        max_distance = int(similarity_th * frame.iloc[0]['cdr3_aa_length'])

        vgene = frame['v_gene'].iloc[0]
        jgene = frame['j_gene'].iloc[0]
        cdr3_aa_length = frame['cdr3_aa_length'].iloc[0]

        selected_clusters.loc[len(selected_clusters), :] = [cluster_id, cluster_center, max_distance, vgene, jgene,
                                                            cdr3_aa_length]

    print('{} out of {} clusters passed the filtering after {} msec'.format(len(selected_clusters),
                                                                            len(data_file[cluster_id_column].unique()),
                                                                            time.time() - t0))

    return selected_clusters


def find_clusters(data_file: pd.DataFrame, vectors: np.array, cluster_id_column, similarity_th):

    data_file[cluster_id_column] = None

    vectors_id = ray.put(vectors)

    data_file['v_gene'] = data_file['v_call'].str.split('-').apply(lambda x: x[0]).str.split('*').apply(
        lambda x: x[0])
    data_file['j_gene'] = data_file['j_call'].str.split('-').apply(lambda x: x[0]).str.split('*').apply(
        lambda x: x[0])

    results_ids = []
    cluster_max_id = 0

    sequences_completed = 0
    for agg_idx, frame in data_file.groupby(['v_gene', 'j_gene', 'cdr3_aa_length']):

        if len(frame) == 1:
            data_file.loc[frame.index, cluster_id_column] = cluster_max_id
            cluster_max_id += 1
            sequences_completed += 1
            continue

        frame_distance_th = int(similarity_th * agg_idx[2])
        results_ids += [(frame.index, do_agglomerative_clustering.remote(vectors_id,
                                                                         frame.index.tolist(),
                                                                         frame_distance_th))]
    t0 = time.time()
    for (frame_index, res_id) in results_ids:

        auto_garbage_collect()

        cluster_labels = np.copy(ray.get(res_id))
        cluster_labels += cluster_max_id
        cluster_max_id += len(cluster_labels)
        data_file.loc[frame_index, cluster_id_column] = cluster_labels

        sequences_completed += len(frame_index)

        print('{:.2f}% of sequences completed after {} msec'.format(sequences_completed * 100 / len(data_file),
                                                                    time.time() - t0))

    print('found {} clusters'.format(len(data_file[cluster_id_column].unique())))
    return data_file


def auto_garbage_collect(pct=80.0):
    if psutil.virtual_memory().percent >= pct:
        gc.collect()


@ray.remote
def do_agglomerative_clustering(vectors: np.array, frame_index: list, distance_threshold):

    frame_vectors = vectors[frame_index, :]
    clustering = AgglomerativeClustering(affinity='manhattan',
                                         linkage='complete',
                                         distance_threshold=distance_threshold,
                                         n_clusters=None).fit(frame_vectors)
    return clustering.labels_


if __name__ == '__main__':
    main()




