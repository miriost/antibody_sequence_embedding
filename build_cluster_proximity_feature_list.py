import pandas as pd
import numpy as np
import sys, argparse
import os
import ast
import json
import ray


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('data_file_path', help='the tsv file path')
    parser.add_argument('output_folder_path', help='Output folder for the feature list file')
    parser.add_argument('output_description', help='description to use inside output file names')
    parser.add_argument('--labels',
                        help='semicolon separated list of labels from which derive cluster significance, '
                             'default is all labels.')
    parser.add_argument('--max_distance', help='max allowed distance in cluster, default is 999', type=float,
                        default=1.0)
    parser.add_argument('--min_subjects', help='minimal number of subjects for cluster selection, default is 5',
                        type=int, default=7)
    parser.add_argument('--min_significance', help='minimal significance of label for cluster selection, default is',
                        type=int, default=0.7)
    parser.add_argument('--cpus', help='How many cores to run in parallel -  default is 1.', type=int, default=1)

    args = parser.parse_args()

    label_column = 'subject.disease_diagnosis'
    id_column = 'subject.subject_id'

    if not (os.path.isfile(args.data_file_path)):
        print('data file error, make sure file path exists\nExiting...')
        sys.exit(1)

    # load files
    data_file = pd.read_csv(args.data_file_path, sep='\t')
    if label_column not in data_file.columns:
        print("{} is not in data file columns: {}\nExisting...".format(label_column, data_file.columns))
        sys.exit(1)

    ray.init()

    if args.labels is None:
        labels = data_file[label_column].unique().tolist()
    else:
        labels = args.labels.split(';')

    subjects = data_file.loc[:, ].groupby(by=[id_column])[label_column].apply(lambda x: x.iloc[0])
    max_distance = args.max_distance

    cpus = args.cpus
    step = data_file.shape[0] / cpus
    ranges = [[round(step * i), round(step * (i + 1))] for i in range(cpus)]

    result_ids = []
    for sub_range in ranges:
        data = data_file.iloc[sub_range[0]:sub_range[1], :]
        result_ids += [build_maps.remote(data, subjects, labels, max_distance)]

    cluster_neighbors_list = []
    data_list = []
    for result_id in result_ids:
        data, cluster_neighbors = ray.get(result_id)
        data_list += [data]
        cluster_neighbors_list += [cluster_neighbors]

    data_file = pd.concat(data_list)
    cluster_neighbors = np.concatenate(cluster_neighbors_list)

    # processing
    number_of_features_neto = 0
    number_of_features_bruto = 0
    # A list with the same length as the data, nan on all cells except the chosen features, which will have 1
    selected_feature_indexes = [np.nan] * len(data_file)
    # A list with the same length as the data, nan on all cells except the chosen points, which will have the index of
    # the original cell which created the cluster
    neighbors_feature_index = [np.nan] * len(data_file)

    min_significance = args.min_significance
    min_subjects = args.min_subjects

    for label in labels:

        candidates_pool = data_file[data_file['how_many_subjects'] >= min_subjects]
        candidates_pool = candidates_pool[data_file[label] >= min_significance]
        candidates_pool = candidates_pool.sort_values(by=[label, 'how_many_subjects'], ascending=[False, False])

        number_of_feature_labels = 0
        for idx, val in candidates_pool.iterrows():
            number_of_features_bruto += 1
            if np.isnan(neighbors_feature_index[idx]):
                number_of_feature_labels += 1
                number_of_features_neto += 1
                selected_feature_indexes[idx] = 1
                for neighbor in cluster_neighbors[idx]:
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

    features_df = data_file.iloc[selected_features, :]

    features_df.to_csv(os.path.join(args.output_folder_path, args.output_description + '.tsv'), sep='\t',
                       index_label='feature_index')
    print('file saved to ', os.path.join(args.output_folder_path, args.output_description + '.tsv'))


@ray.remote
def build_maps(data, subjects, labels, max_distance):
    distance_map = np.array(data['cluster_distances'].apply(lambda x: json.loads(x)).tolist())

    data['cluster_subjects'] = data['cluster_subjects'].apply(lambda x: ast.literal_eval(x))
    data['cluster_neighbors'] = data['cluster_neighbors'].apply(lambda x: ast.literal_eval(x))

    # need to filter neighbors by max_distance
    filtering = distance_map <= max_distance
    cluster_subjects = list(map(lambda i: np.array(data.iloc[i, :]['cluster_subjects'])[filtering[i, :]], range(len(data))))

    # drop None values because of unassigned neighbors due to lack of candidates, reduce to unique subjects
    cluster_subjects = np.array(list(map(lambda x: np.unique(pd.Series(x).dropna().to_list()), cluster_subjects)))

    cluster_neighbors = np.array(list(map(lambda i: np.array(data.iloc[i, :]['cluster_neighbors'])[filtering[i, :]], range(len(data)))),
                                 dtype=object)

    # cluster diameter
    data['max_distance'] = list(map(lambda i: np.max(distance_map[i, filtering[i, :]]), range(distance_map.shape[0])))

    # for analysis - subjects from each label in the clusters
    for label in labels:
        data[label + ' subjects'] = list(map(lambda x: subjects[(subjects.index.isin(x)) & (subjects == label)].index.tolist(),
                                             cluster_subjects))

    # for the cluster selection - what is the proportion of each label in the clusters
    cluster_diagnosis = pd.concat(list(map(lambda x: subjects[x].value_counts(normalize=True, dropna=True), cluster_subjects)),
                                  sort=True, axis=1, ignore_index=True).transpose().fillna(0)
    data[labels] = cluster_diagnosis[labels]

    # for the cluster selection - how many subjects are in the clusters
    data['how_many_subjects'] = list(map(lambda x: len(x), cluster_subjects))

    return data, cluster_neighbors


if __name__ == '__main__':
    main()

