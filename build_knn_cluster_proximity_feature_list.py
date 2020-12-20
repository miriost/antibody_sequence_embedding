import pandas as pd
import numpy as np
import sys, argparse
import os
import ray
import psutil
import gc


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('data_file_path', help='the data tsv file path')
    parser.add_argument('vectors_file_path', help='the vectors npy file path')
    parser.add_argument('distances_file_path', help='the distance npy file path')
    parser.add_argument('neighbors_file_path', help='the neighbors npy file path')
    parser.add_argument('output_folder_path', help='Output folder for the feature list file')
    parser.add_argument('output_description', help='description to use inside output file names')
    parser.add_argument('--labels',
                        help='semicolon separated list of labels from which derive cluster significance, '
                             'default is all labels.')
    parser.add_argument('--max_distance', help='max allowed distance in cluster, default is the 15 percentile of the '
                                               'knn distances', type=float)
    parser.add_argument('--max_distance_percentile', help='percentile to determine max_distance if max_distance is not '
                                                          'provided', type=int, default=15)
    parser.add_argument('--min_subjects', help='minimal number of subjects for cluster selection, default is 7',
                        type=int, default=7)
    parser.add_argument('--min_significance', help='minimal significance of label for cluster selection, default is 0.7',
                        type=float, default=0.7)
    parser.add_argument('--num_cpus', help='How many cores to run in parallel -  default is psutil.cpu_count()',
                        type=int)
    parser.add_argument('--shuffle_seed', help='shuffle the subjects labels before creating the feature list, default '
                                               'is no shuffling.', type=int)
    parser.add_argument('--max_distance_diameter', help='use max distance as the cluster diameter, default is False',
                        type=str2bool, default=False)
    parser.add_argument('--use_centroid', help='calculate cluster centroid as an average of all members coordinates',
                        type=str2bool, default=False)

    args = parser.parse_args()

    label_column = 'subject.disease_diagnosis'
    id_column = 'subject.subject_id'
    min_significance = args.min_significance
    min_subjects = args.min_subjects
    num_cpus = args.num_cpus
    shuffle_seed = args.shuffle_seed
    max_distance = args.max_distance
    max_distance_diameter = args.max_distance_diameter
    use_centroid = args.use_centroid
    data_file_path = args.data_file_path
    vectors_file_path = args.vectors_file_path
    distances_file_path = args.distances_file_path
    neighbors_file_path = args.neighbors_file_path
    labels = args.labels
    max_distance_percentile = args.max_distance_percentile

    if num_cpus is None:
        num_cpus = psutil.cpu_count()

    if not (os.path.isfile(data_file_path)):
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
    print('loaded data file')

    vectors_file = np.load(vectors_file_path)
    if vectors_file.shape[0] != data_file.shape[0]:
        print('mismatch between data_file and vectors_file length\nExiting...')
        sys.exit(1)
    print('loaded vectors file')

    distances_file = np.load(distances_file_path)
    if distances_file.shape[0] != data_file.shape[0]:
        print('mismatch between data_file and distances_file length\nExiting...')
        sys.exit(1)
    print('loaded distances file')

    neighbors_file = np.load(neighbors_file_path)
    if neighbors_file.shape[0] != data_file.shape[0]:
        print('mismatch between data_file and neighbors_file length\nExiting...')
        sys.exit(1)
    print('loaded neighbors file')

    ray.init(num_cpus=num_cpus)

    if labels is None:
        labels = data_file[label_column].unique().tolist()
    else:
        labels = args.labels.split(';')

    if max_distance is None:
        max_distance = np.percentile(distances_file, max_distance_percentile)
        print('max_distance {} (percentile {})'.format(max_distance, max_distance_percentile))

    subjects = data_file.loc[:, ].groupby(by=[id_column])[label_column].apply(lambda x: x.iloc[0])
    if shuffle_seed is not None:
        print("shuffling labels before building feature list")
        subjects[:] = np.random.RandomState(seed=shuffle_seed).permutation(subjects)

    step = data_file.shape[0] / num_cpus
    ranges = [[round(step * i), round(step * (i + 1))] for i in range(num_cpus)]

    data_file_id = ray.put(data_file)
    distances_file_id = ray.put(distances_file)
    neighbors_file_id = ray.put(neighbors_file)

    data_file = pd.concat(ray.get([build_maps.remote(data_file_id,
                                                     data_file[sub_range[0]:sub_range[1]],
                                                     distances_file_id,
                                                     neighbors_file_id,
                                                     sub_range,
                                                     subjects,
                                                     labels,
                                                     max_distance) for sub_range in ranges]))
    if psutil.virtual_memory().percent >= 80.0:
        gc.collect()

    data_file[labels + ['how_many_subjects']].to_csv(os.path.join(args.output_folder_path, args.output_description +
                                                                  '_clusters.tsv'), sep='\t', index=False)

    # processing
    number_of_features_neto = 0
    number_of_features_bruto = 0
    # A list with the same length as the data, nan on all cells except the chosen features, which will have 1
    selected_feature_indexes = [np.nan] * len(data_file)
    # A list with the same length as the data, nan on all cells except the chosen points, which will have the index of
    # the original cell which created the cluster
    neighbors_feature_index = [np.nan] * len(data_file)

    for label in labels:

        candidates_pool = data_file[data_file['how_many_subjects'] >= min_subjects]
        candidates_pool = candidates_pool[candidates_pool[label] >= min_significance]
        candidates_pool = candidates_pool.sort_values(by=[label, 'how_many_subjects', 'max_distance'],
                                                      ascending=[False, False, True])

        print('label {} candidate pool size is {}'.format(label, len(candidates_pool)))

        number_of_feature_labels = 0
        for idx, val in candidates_pool.iterrows():
            number_of_features_bruto += 1
            if np.isnan(neighbors_feature_index[idx]):
                number_of_feature_labels += 1
                number_of_features_neto += 1
                selected_feature_indexes[idx] = 1
                for neighbor in candidates_pool.loc[idx, 'cluster_neighbors']:
                    if neighbor < len(data_file):
                        neighbors_feature_index[neighbor] = idx

        print('Label {}, min subjects {}, min_significance {}, added features {}'.format(
            label, min_subjects, min_significance, number_of_feature_labels))

    print('Number of features that meet the TH criteria:' + str(number_of_features_bruto))
    print('Number of features after filtration of those appeared as neighbors: ' + str(number_of_features_neto))

    # ====
    # build a feature list file, each raw contains feature center, and maximal radius
    selected_features = np.nonzero(~np.isnan(selected_feature_indexes))[0]
    print('selected features: ', selected_features)
    if len(selected_features) != number_of_features_neto:
        print('Error! selected features number mismatch!!')
        exit(1)

    features_df = data_file.iloc[selected_features, :].copy(deep=True)
    features_df.loc[selected_features, 'cluster_sequences'] = data_file.loc[selected_features,
                                                                            'cluster_neighbors'].apply(lambda x: data_file.loc[x, 'document._id'].to_list())

    if use_centroid:
        features_df.loc[selected_features, 'vector'] = features_df['cluster_neighbors'].apply(lambda x: np.mean(vectors_file[x, :], axis=0).tolist())
    else:
        features_df.loc[selected_features, 'vector'] = pd.Series(list(map(lambda x: x.tolist(),
                                                                      vectors_file[selected_features, :])), index=features_df.index)

    if max_distance_diameter:
        features_df['max_distance'] = max_distance

    features_df.to_csv(os.path.join(args.output_folder_path, args.output_description + '.tsv'), sep='\t',
                       index_label='feature_index')
    print('file saved to ', os.path.join(args.output_folder_path, args.output_description + '.tsv'))


@ray.remote
def build_maps(data_file: pd.DataFrame, data: pd.DataFrame, distances_file: np.array, neighbors_file: np.array,
               sub_range, subjects: pd.Series, labels, max_distance):
    distances_map = distances_file[sub_range[0]:sub_range[1], :]
    neighbors_map = neighbors_file[sub_range[0]:sub_range[1], :]

    # need to filter neighbors by max_distance
    filtering = np.logical_and(distances_map <= max_distance, neighbors_map < len(data_file))

    print("adding cluster_distances column to rows {}".format(sub_range))
    cluster_distances = list(map(lambda i: distances_map[i, filtering[i, :]].tolist(), range(len(data))))
    data['cluster_distances'] = pd.Series(cluster_distances, index=data.index)
    del cluster_distances
    del distances_map

    print("adding cluster_neighbors column to rows {}".format(sub_range))
    cluster_neighbors = list(map(lambda i: neighbors_map[i, filtering[i, :]].tolist(), range(len(data))))
    data.loc[data.index, 'cluster_neighbors'] = pd.Series(cluster_neighbors, index=data.index)
    del cluster_neighbors
    del neighbors_map

    del filtering

    print("adding cluster_subjects column to rows {}".format(sub_range))
    data.loc[data.index, 'cluster_subjects'] = data.loc[data.index,
                                                        'cluster_neighbors'].apply(lambda x: data_file.loc[x, 'subject.subject_id'].to_list())

    # cluster diameter
    data['max_distance'] = data['cluster_distances'].apply(lambda x: np.max(x))
    data['mean_distance'] = data['cluster_distances'].apply(lambda x: np.mean(x))

    # for analysis - subjects from each label in the clusters
    print("adding label subjects columns to rows {}".format(sub_range))
    for label in labels:
        data[label + ' subjects'] = data['cluster_subjects'].apply(lambda x: subjects[(subjects.index.isin(x)) & (subjects == label)].index.tolist())

    # for the cluster selection - what is the proportion of each label in the clusters
    print("adding label significant columns to rows {}".format(sub_range))
    cluster_diagnosis = data['cluster_subjects'].apply(lambda x: subjects[x].value_counts(normalize=True,
                                                                                          dropna=True)).fillna(0)
    data[labels] = cluster_diagnosis[labels]
    del cluster_diagnosis

    # for the cluster selection - how many subjects are in the clusters
    data['how_many_subjects'] = data['cluster_subjects'].apply(lambda x: len(x))

    print("finished adding columns to rows {}".format(sub_range))

    return data


if __name__ == '__main__':
    main()



    label_column = 'subject.disease_diagnosis'
    id_column = 'subject.subject_id'
    min_significance = 0.7
    min_subjects = 7
    num_cpus = 1
    shuffle_seed = None
    max_distance = None
    max_distance_diameter = False
    data_file_path = 'data/FOLD0/celiac_bcr_light_chain_db_celiac_bcr_light_chain_prot2vec_10dim_FILTERED_TRAIN_0.tsv'
    vectors_file_path = 'data/FOLD0/celiac_bcr_light_chain_db_celiac_bcr_light_chain_prot2vec_10dim_TRAIN_VECTORS.npy'
    distances_file_path = 'data/FOLD0/10dim_200knn_distances.npy'
    neighbors_file_path = 'data/FOLD0/10dim_200knn_neighbors.npy'
    labels = None