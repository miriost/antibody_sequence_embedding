# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 10:18:13 2020

@author: mirio
"""
"""
Cluster proximity to feature table

Feature selection
================
Start by feature selection according to the following parameters:
    1. min_number_of_subjects- how many subjects are in each cluster
    2. Significance - only clusters with higher score will be chosen abs(1-score) 
    * We will want to eliminate clusters that appeared already
    
"""
import pandas as pd
import numpy as np
import json
import sys, argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file_path',
                        help='the tsv file path')
    parser.add_argument('--analysis_file_path',
                        help='the analysis file path')
    parser.add_argument('--distances_file_path',
                        help='the distances file path')
    parser.add_argument('--knn_file_path',
                        help='the knn_file file path')
    parser.add_argument('--label_column',
                        help='the name of the column with the labels. Default is repertoire.disease_diagnosis',
                        default='repertoire.disease_diagnosis')
    parser.add_argument('--id_column',
                        help='the name of the column with the subject id. Default is repertoire.repertoire_name',
                        default='repertoire.repertoire_name')
    parser.add_argument('--output_folder_path',
                        help='Output folder for the 3 output files - Nearest neighbors file, distances file, and '
                             'results analysis file')
    parser.add_argument('--output_description',
                        help='description to use inside output file names')
    parser.add_argument('--labels',
                        help='semicolon separated list of labels from which derive cluster significance')
    parser.add_argument('--min_subjects', help='minimal number of subjects for cluster selection', type=int)
    parser.add_argument('--max_features', help='max selected features per label', default=100, type=int)

    args = parser.parse_args()

    if not (os.path.isfile(args.data_file_path)):
        print('data file error, make sure file path exists\nExiting...')
        sys.exit(1)

    if not (os.path.isfile(args.analysis_file_path)):
        print('analysis file error, make sure file path exists\nExiting...')
        sys.exit(1)

    if not (os.path.isfile(args.distances_file_path)):
        print('distances file error, make sure file path exists\nExiting...')
        sys.exit(1)

    if not (os.path.isfile(args.knn_file_path)):
        print('distances file error, make sure file path exists\nExiting...')
        sys.exit(1)

    if args.labels is None:
        print("Missing labels argument\nExisting...")
        sys.exit(1)
    labels = args.labels.split(';')

    # load files
    data_file = pd.read_csv(args.data_file_path, sep='\t')
    if args.label_column not in data_file.columns:
        print("{} is not in data file columns: {}\nExisting...".format(args.label_column, data_file.columns))
        sys.exit(1)

    if args.id_column not in data_file.columns:
        print("{} is not in data file columns: {}\nExisting...".format(args.id_column, data_file.columns))
        sys.exit(1)

    analysis_file = pd.read_csv(args.analysis_file_path)
    for label in labels:
        if label not in analysis_file.columns:
            print("{} is not in data file columns: {}\nExisting...".format(label, analysis_file.columns))
            sys.exit(1)

    distance_file = pd.read_csv(args.distances_file_path)

    knn_file = pd.read_csv(args.knn_file_path, sep='\t')
    vectors = np.array(knn_file['vector'].apply(lambda x: json.loads(x)).to_list())
    vectors = pd.DataFrame(vectors, columns=list(range(vectors.shape[1])))

    # processing
    number_of_features_neto = 0
    number_of_features_bruto = 0
    # A list with the same length as the data, nan on all cells except the chosen features, which will have 1
    selected_feature_indexes = [np.nan] * len(analysis_file)
    # A list with the same length as the data, nan on all cells except the chosen points, which will have the index of
    # the original cell which created the cluster
    neighbors_feature_index = [np.nan] * len(analysis_file)

    for label in labels:
        num_of_subjects_in_label = len(data_file.loc[data_file[args.label_column]==label, args.id_column].unique())
        if args.min_subjects is None:
            min_subjects = min(num_of_subjects_in_label, 5)
        else:
            min_subjects = min(num_of_subjects_in_label, args.min_subjects)

        candidates_pool = analysis_file[analysis_file['how_many_subjects'] >= min_subjects]
        candidates_pool = candidates_pool.sort_values(by=[label, 'how_many_subjects'], ascending=False)

        number_of_feature_labels = 0
        for idx, val in candidates_pool.iterrows():
            number_of_features_bruto += 1
            if np.isnan(neighbors_feature_index[ idx ]):
                number_of_feature_labels += 1
                number_of_features_neto += 1
                selected_feature_indexes[ idx ] = 1
                for neighbor in map(int, val[ 'neighbors' ][ 1:-1 ].split(',')):
                    neighbors_feature_index[ neighbor ] = idx
                if number_of_feature_labels == args.max_features:
                    break

        print('Label {}, min subjects {}, max_features {}, added features {}'.format(
            label, min_subjects, args.max_features, number_of_feature_labels))

    print('Number of features that meet the TH criteria:' + str(number_of_features_bruto))
    print('Number of features after filtration of those appeared as neighbors: ' + str(number_of_features_neto))

    # ====
    # build a feature list file, each raw contains feature center, and maximal radius
    c = ['feature_index', 'max_distance']
    c.extend(vectors.columns)
    selected_features = np.nonzero(~np.isnan(selected_feature_indexes))[0]
    print('selected features: ', selected_features)
    if len(selected_features) != number_of_features_neto:
        print('Error! selected features number mismatch!!')
    else:
        print(f'Beginning building feature list with {number_of_features_neto} features')

    features_df = pd.DataFrame(0, index=list(range(0, number_of_features_neto)), columns=c)
    features_df['feature_index'] = selected_features
    features_df['max_distance'] = distance_file.iloc[ selected_features ].max(axis=1).values
    features_df.iloc[:, 2:] = vectors.iloc[ selected_features ].values

    features_df.to_csv(os.path.join(args.output_folder_path, args.output_description + '.csv'), index=False)
    print('file saved to ', os.path.join(args.output_folder_path, args.output_description + '.csv'))


if __name__ == '__main__':
    main()

