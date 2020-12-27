import argparse
import sys, os
import pandas as pd
import numpy as np
from scipy.stats import mode


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('data_file_path', help='the filtered data file path')
    parser.add_argument('vectors_file_path', help='the vectors file path')
    parser.add_argument('output_folder_path', help='Output folder for the feature list file')
    parser.add_argument('output_description', help='description to use inside output file names')
    parser.add_argument('cluster_id_column', help='the name of the column with the cluster_id')
    parser.add_argument('--subjects_th', help='th filter by min subjects, default is 5', default=5, type=int)
    parser.add_argument('--significance_th', help='th filter by min significance, default is 0.75', default=0.75, type=float)
    parser.add_argument('--is_manhattan', help='use cdr3_aa bytes array as vectors, default is False', type=str2bool, default=False)
    parser.add_argument('--max_distance_th', help='The distance diameter for the cluster', type=float, default=0.15)

    args = parser.parse_args()

    data_file_path = args.data_file_path
    vectors_file_path = args.vectors_file_path
    subjects_th = args.subjects_th
    significance_th = args.significance_th
    is_manhattan = args.is_manhattan
    output_folder_path = args.output_folder_path
    output_description = args.output_description
    cluster_id_column = args.cluster_id_column
    max_distance_th = args.max_distance_th

    if not os.path.isfile(data_file_path):
        print('Data file error, make sure data file path: {}\nExiting...'.format(data_file_path))
        sys.exit(1)

    if not os.path.isfile(vectors_file_path):
        print('Data file error, make sure vectors file path: {}\nExiting...'.format(vectors_file_path))
        sys.exit(1)

    data_file = pd.read_csv(data_file_path, sep='\t')
    if cluster_id_column not in data_file.columns:
        print('{} is not in data_file columns: {}\nExiting...'.format(cluster_id_column, data_file.columns))
        sys.exit(1)

    print('loaded data file')

    vectors_file = np.load(vectors_file_path)
    if vectors_file.shape[0] != data_file.shape[0]:
        print('mismatch between data_file and vectors_file length\nExiting...')
        sys.exit(1)
    print('loaded vectors file')

    selected_clusters = pd.DataFrame(columns=['feature_index', 'vector', 'max_distance'])

    for cluster_id, frame in data_file.groupby([cluster_id_column]):

        if len(frame['subject.subject_id'].unique()) < subjects_th:
            continue

        frame_value_counts = frame['subject.disease_diagnosis'].value_counts(normalize=True)
        if frame_value_counts.max() < significance_th:
            continue

        if is_manhattan is True:
            frame_vectors = np.array(frame['cdr3_aa'].apply(lambda x: [b for b in 'ab'.encode('utf-8')]).to_list())
            cluster_center = mode(frame_vectors)[0].tolist()
            max_distance = int(max_distance_th * frame.iloc[0]['cdr3_aa_length'])
        else:
            frame_vectors = vectors_file[frame.index, :]
            cluster_center = np.mean(frame_vectors, axis=0).tolist()
            max_distance = max_distance_th

        selected_clusters.loc[len(selected_clusters), :] = [cluster_id, cluster_center, max_distance]

    print('{} out of {} clusters passed the filtering'.format(len(selected_clusters),
                                                              len(data_file['cluster_id'].unique())))

    selected_clusters.to_csv(os.path.join(output_folder_path, output_description + '.tsv'), sep='\t',
                             index_label='feature_index')
    print('output is saved to {}'.format(os.path.join(output_folder_path, output_description + '.tsv')))


if __name__ == '__main__':
    main()