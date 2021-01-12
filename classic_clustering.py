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
from sklearn.metrics.pairwise import pairwise_distances


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('train_file_path', help='The preprocessed train data file path')
    parser.add_argument('output_desc', help='Description prefix for the output file names')
    parser.add_argument('output_folder_path', help='Output folder for the output knn files')
    parser.add_argument('subject_col', help='name of the column with the subjects ids')
    parser.add_argument('label_col', help='name of the column with the disease diagnosis')

    parser.add_argument('--test_file_path', help='The preprocessed test data file path')
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

    train_file_path = args.train_file_path
    if not os.path.isfile(train_file_path):
        print('Data file error, make sure data file path: {}\nExiting...'.format(train_file_path))
        sys.exit(1)

    test_file_path = args.test_file_path
    if test_file_path is not None and not os.path.isfile(test_file_path):
        print('Data file error, make sure data file path: {}\nExiting...'.format(train_file_path))
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

    train_data_file = pd.read_csv(train_file_path, sep='\t')
    print('loaded data file')

    train_data_file = find_clusters(train_data_file, cluster_id_column, similarity_th)
    train_data_file.to_csv(train_file_path, sep='\t', index=False)

    selected_clusters = filter_clusters(train_data_file,
                                        cluster_id_column, subject_col, label_col,
                                        similarity_th, subjects_th, significance_th)

    selected_clusters.to_csv(os.path.join(output_folder_path, output_desc + '_features.tsv'), sep='\t', index=False)
    print('feature list is saved to {}'.format(os.path.join(output_folder_path, output_desc + '.tsv')))

    train_feature_table = build_feature_table(train_data_file, selected_clusters, subject_col)
    train_feature_table.to_csv(os.path.join(output_folder_path, output_desc + '_train.tsv'), sep='\t', index=False)
    print('train feature table is saved to {}'.format(os.path.join(output_folder_path, output_desc + '_train.tsv')))

    if test_file_path is not None:
        test_data_file = pd.read_csv(test_file_path, sep='\t')
        test_feature_table = build_feature_table(test_data_file, selected_clusters, subject_col)
        test_feature_table.to_csv(os.path.join(output_folder_path, output_desc + '_test.tsv'), sep='\t', index=False)
        print('test feature table is saved to {}'.format(os.path.join(output_folder_path, output_desc + '_test.tsv')))


def filter_clusters(data_file: pd.DataFrame, cluster_id_column, subject_col, label_col,
                    similarity_th, subjects_th, significance_th) -> pd.DataFrame:

    selected_clusters = pd.DataFrame(columns=['cluster_id', 'center', 'max_distance', 'v_gene', 'j_gene',
                                              'cdr3_aa_length', 'num_subjects', 'label', 'significance'])

    # filter clusters which has no chance to pass filter (has less than subjects_th sequences)
    clusters = data_file[cluster_id_column].value_counts()
    clusters = clusters[clusters > subjects_th]
    data_file = data_file[data_file[cluster_id_column].isin(clusters.keys().tolist())]

    t0 = time.time()
    for cluster_id, frame in data_file.groupby([cluster_id_column]):

        frame_subject_counts = frame[subject_col].value_counts(normalize=True)
        num_subjects = len(frame_subject_counts)
        if num_subjects < subjects_th:
            continue

        if frame_subject_counts.values[0] >= 0.9:
            # if more than 90% of the sequences come from the same subject - skip
            continue

        frame_value_counts = frame[label_col].value_counts(normalize=True)
        significance = frame_value_counts.values[0]
        if significance < significance_th:
            continue
        label = frame_value_counts.index[0]

        frame_vectors = np.array(frame['cdr3_aa'].apply(lambda x: [b for b in x.encode('utf-8')]).to_list())
        cluster_center = mode(frame_vectors)[0][0].tolist()
        cluster_center = ''.join(chr(c) for c in cluster_center)
        max_distance = int(similarity_th * frame.iloc[0]['cdr3_aa_length'])

        vgene = frame['v_gene'].iloc[0]
        jgene = frame['j_gene'].iloc[0]
        cdr3_aa_length = frame['cdr3_aa_length'].iloc[0]

        selected_clusters.loc[len(selected_clusters), :] = [cluster_id, cluster_center, max_distance, vgene, jgene,
                                                            cdr3_aa_length, num_subjects, label, significance]

    print('{} out of {} clusters passed the filtering after {} msec'.format(len(selected_clusters),
                                                                            len(data_file[cluster_id_column].unique()),
                                                                            time.time() - t0))

    return selected_clusters


def find_clusters(data_file: pd.DataFrame, cluster_id_column, similarity_th) -> pd.DataFrame:

    data_file[cluster_id_column] = None

    data_file['v_gene'] = data_file['v_call'].str.split('-').apply(lambda x: x[0]).str.split('*').apply(
        lambda x: x[0])
    data_file['j_gene'] = data_file['j_call'].str.split('-').apply(lambda x: x[0]).str.split('*').apply(
        lambda x: x[0])

    results_ids = []
    cluster_max_id = 0

    sequences_completed = 0

    t0 = time.time()

    for agg_idx, frame in data_file.groupby(['v_gene', 'j_gene', 'cdr3_aa_length']):

        if len(frame) == 1:
            data_file.loc[frame.index, cluster_id_column] = cluster_max_id
            cluster_max_id += 1
            sequences_completed += 1
            continue

        frame_vectors = np.array(frame['cdr3_aa'].apply(lambda x: [b for b in x.encode('utf-8')]).to_list())
        frame_distance_th = int(similarity_th * agg_idx[2])
        results_ids += [(frame.index, do_agglomerative_clustering.remote(frame_vectors, frame_distance_th))]

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
def do_agglomerative_clustering(vectors: np.array, distance_threshold):

    clustering = AgglomerativeClustering(affinity='manhattan',
                                         linkage='complete',
                                         distance_threshold=distance_threshold,
                                         n_clusters=None).fit(vectors)
    return clustering.labels_


def build_feature_table(data_file: pd.DataFrame, features_file: pd.DataFrame, subject_col: str) -> pd.DataFrame:

    data_file_id = ray.put(data_file)
    features_file_id = ray.put(features_file)

    result_ids = []
    for subject, frame in data_file.groupby(subject_col):
        result_ids += [get_subject_feature_table.remote(subject, data_file_id, features_file_id, subject_col)]

    features_table = pd.concat([ray.get(res_id) for res_id in result_ids])

    return features_table


@ray.remote
def get_subject_feature_table(subject: str, data_file: pd.DataFrame, features_file: pd.DataFrame,
                              subject_col: str) -> pd.DataFrame:

    t0 = time.time()

    print('Start creating feature table for subject {}'.format(subject))

    subject_data = data_file[data_file[subject_col] == subject]
    subject_features_table = pd.DataFrame(0, index=[subject], columns=features_file['cluster_id'].values)
    features_count = 0

    by = ['v_gene', 'j_gene', 'cdr3_aa_length']
    for agg_idx, feature_frame in features_file.groupby(by):

        subject_frame = subject_data[(subject_data['v_gene'] == agg_idx[0]) &
                                     (subject_data['j_gene'] == agg_idx[1]) &
                                     (subject_data['cdr3_aa_length'] == agg_idx[2])]

        if len(subject_frame) == 0:
            continue

        subject_vectors = np.array(subject_frame['cdr3_aa'].apply(lambda x: [b for b in x.encode('utf-8')]).to_list())
        feature_vectors = np.array(feature_frame['center'].apply(lambda x: [b for b in x.encode('utf-8')]).to_list())

        # for each vector belonging to the subject
        distances = pairwise_distances(X=subject_vectors, Y=feature_vectors, metric='manhattan')
        distance_close_enough_mat = np.less_equal(distances, features_file['max_distance'].to_numpy())

        features_count += np.sum(distance_close_enough_mat)
        if features_count >= 1:
            for distance_close_enough_vec in distance_close_enough_mat:
                if np.sum(distance_close_enough_vec) == 0:
                    continue
                cluster_id = np.where(distance_close_enough_vec)
                subject_features_table.loc[subject, feature_frame['cluster_id'].iloc[cluster_id[0]]] += 1

    print('Finished creating feature table for subject {}, non zero feature count {}, took {}'.format(
        subject, features_count, time.time() - t0))

    return subject_features_table


if __name__ == '__main__':
    main()




