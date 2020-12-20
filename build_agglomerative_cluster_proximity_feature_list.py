from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import numpy as np
import sys, argparse
import os
from scipy.sparse import csr_matrix


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('data_file_path', help='the data tsv file path')
    parser.add_argument('vectors_file_path', help='the vectors npy file path')
    parser.add_argument('output_folder_path', help='Output folder for the feature list file')
    parser.add_argument('output_description', help='description to use inside output file names')
    parser.add_argument('--neighbors_file_path', help='the neighbors npy file path')
    parser.add_argument('--distances_file_path', help='the distance npy file path')
    parser.add_argument('--labels',
                        help='semicolon separated list of labels from which derive cluster significance, '
                             'default is all labels.')
    parser.add_argument('--min_subjects', help='minimal number of subjects for cluster selection, default is 7',
                        type=int, default=7)
    parser.add_argument('--min_significance', help='minimal significance of label for cluster selection, default is 0.7',
                        type=float, default=0.7)
    parser.add_argument('--shuffle_seed', help='shuffle the subjects labels before creating the feature list, default '
                                               'is no shuffling.', type=int)

    args = parser.parse_args()

    label_column = 'subject.disease_diagnosis'
    id_column = 'subject.subject_id'
    min_significance = args.min_significance
    min_subjects = args.min_subjects
    shuffle_seed = args.shuffle_seed
    data_file_path = args.data_file_path
    vectors_file_path = args.vectors_file_path
    distances_file_path = args.distances_file_path
    neighbors_file_path = args.neighbors_file_path

    if not os.path.isfile(data_file_path):
        print('data file error, make sure file path exists\nExiting...')
        sys.exit(1)

    if not (os.path.isfile(vectors_file_path)):
        print('vectors file error, make sure file path exists\nExiting...')
        sys.exit(1)

    # load files
    data_file = pd.read_csv(data_file_path, sep='\t')
    if label_column not in data_file.columns:
        print("{} is not in data file columns: {}\nExisting...".format(label_column, data_file.columns))
        sys.exit(1)

    if args.labels is None:
        labels = data_file[label_column].unique().tolist()
    else:
        labels = args.labels.split(';')

    vectors_file = np.load(vectors_file_path)
    if vectors_file.shape[0] != data_file.shape[0]:
        print('mismatch between data_file and vectors_file length\nExiting...')
        sys.exit(1)

    distance_threshold = 0.15
    if distances_file_path is not None:
        distances_file = np.load(distances_file_path)
        if distances_file.shape[0] != data_file.shape[0]:
            print('mismatch between data_file and distances_file length\nExiting...')
            sys.exit(1)

        distance_threshold = np.percentile(distances_file, q=25)

    if neighbors_file_path is not None:
        neighbors_file = np.load(neighbors_file_path)
        if neighbors_file.shape[0] != data_file.shape[0]:
            print('mismatch between data_file and neighbors_file length\nExiting...')
            sys.exit(1)

        if distances_file_path is not None and neighbors_file.shape[1] != distances_file.shape[1]:
            print('mismatch between neighbors_file and distances_file number of columns\nExiting...')
            sys.exit(1)

        data = np.ones([neighbors_file.size])
        cols = neighbors_file.reshape([neighbors_file.size])
        rows = np.repeat(np.arange(neighbors_file.shape[0]), repeats=neighbors_file.shape[1])
        connectivity = csr_matrix((data, (rows, cols)), shape=(neighbors_file.shape[0], neighbors_file.shape[0]))

        del data
        del rows
        del cols

        clustering = AgglomerativeClustering(connectivity=connectivity,
                                             affinity='euclidean',
                                             linkage='complete',
                                             distance_threshold=distance_threshold,
                                             n_clusters=None).fit(vectors_file)

    else:
        clustering = AgglomerativeClustering(affinity='euclidean',
                                             linkage='complete',
                                             distance_threshold=distance_threshold,
                                             n_clusters=None).fit(vectors_file)

    data_file['cluster_id'] = clustering.labels_
    data_file.to_csv('data_file_path', sep='\t', index=False)

    subjects = data_file.loc[:, ].groupby(by=[id_column])[label_column].apply(lambda x: x.iloc[0])
    if shuffle_seed is not None:
        print("shuffling labels before building feature list")
        subjects[:] = np.random.RandomState(seed=shuffle_seed).permutation(subjects)

    feature_list = pd.DataFrame(columns=labels)
    row_idx = 0
    for cluster_id, frame in data_file.groupby(['cluster_id']):
        cluster_subjects = frame[id_column].unique()
        if len(cluster_subjects) < min_subjects:
            continue
        cluster_labels = subjects[cluster_subjects].value_counts(normalize=True)
        for label in labels:
            if label in cluster_labels.index and cluster_labels[label] > min_significance:
                feature_list.loc[row_idx, cluster_labels.index] = cluster_labels.values
                feature_list.loc[row_idx, [l + '_subjects' for l in cluster_labels.index]] = \
                    [subjects[cluster_subjects][subjects[cluster_subjects]==l].index.to_list() for l in cluster_labels.index]
                row_idx += 1
                break

    feature_list.to_csv('features_list', sep='\t', index=False)


if __name__ == '__main__':
    main()

