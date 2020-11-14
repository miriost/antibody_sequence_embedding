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
    parser.add_argument('--label_column',
                        help='the name of the column with the labels. Default is repertoire.disease_diagnosis',
                        default='repertoire.disease_diagnosis')
    parser.add_argument('--id_column',
                        help='the name of the column with the subject id. Default is repertoire.repertoire_name',
                        default='repertoire.repertoire_name')
    parser.add_argument('--output_folder_path',
                        help='Output folder for the feature list file')
    parser.add_argument('--output_description',
                        help='description to use inside output file names')
    parser.add_argument('--labels',
                        help='semicolon separated list of labels from which derive cluster significance')
    parser.add_argument('--min_subjects', help='minimal number of subjects for cluster selection, default is 5',
                        type=int, default=5)
    parser.add_argument('--max_subjects', help='maximal number of subjects for cluster selection, default is 50',
                        type=int, default=50)
    parser.add_argument('--max_features', help='max selected features per label', default=100, type=int)

    args = parser.parse_args()

    if not (os.path.isfile(args.data_file_path)):
        print('data file error, make sure file path exists\nExiting...')
        sys.exit(1)

    if not (os.path.isfile(args.analysis_file_path)):
        print('analysis file error, make sure file path exists\nExiting...')
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

    # processing
    number_of_features_neto = 0
    number_of_features_bruto = 0
    # A list with the same length as the data, nan on all cells except the chosen features, which will have 1
    selected_feature_indexes = [np.nan] * len(analysis_file)
    # A list with the same length as the data, nan on all cells except the chosen points, which will have the index of
    # the original cell which created the cluster
    neighbors_feature_index = [np.nan] * len(analysis_file)

    min_significance = 0.6
    min_subjects = args.min_subjects
    max_subjects = args.max_subjects

    for label in labels:

        candidates_pool = analysis_file[analysis_file['how_many_subjects'] >= min_subjects]
        candidates_pool = candidates_pool[analysis_file['how_many_subjects'] <= max_subjects]
        min_significance = min(min_significance, candidates_pool[label].max())
        candidates_pool = candidates_pool[analysis_file[label] >= min_significance]
        candidates_pool = candidates_pool.sort_values(by=[label, 'how_many_subjects'],
                                                      ascending=[False, False])

        number_of_feature_labels = 0
        for idx, val in candidates_pool.iterrows():
            number_of_features_bruto += 1
            if np.isnan(neighbors_feature_index[idx]):
                number_of_feature_labels += 1
                number_of_features_neto += 1
                selected_feature_indexes[idx] = 1
                for neighbor in map(int, val['neighbors'][1:-1].split(',')):
                    neighbors_feature_index[neighbor] = idx
                if number_of_feature_labels == args.max_features:
                    break

        print('Label {}, min subjects {}, max_features {}, added features {}'.format(
            label, min_subjects, args.max_features, number_of_feature_labels))

    print('Number of features that meet the TH criteria:' + str(number_of_features_bruto))
    print('Number of features after filtration of those appeared as neighbors: ' + str(number_of_features_neto))

    # ====
    # build a feature list file, each raw contains feature center, and maximal radius
    selected_features = np.nonzero(~np.isnan(selected_feature_indexes))[0]
    print('selected features: ', selected_features)
    if len(selected_features) != number_of_features_neto:
        print('Error! selected features number mismatch!!')
    else:
        print(f'Beginning building feature list with {number_of_features_neto} features')

    features_df = analysis_file.iloc[selected_features, :]

    features_df.to_csv(os.path.join(args.output_folder_path, args.output_description + '.csv'), index_label='feature_index')
    print('file saved to ', os.path.join(args.output_folder_path, args.output_description + '.csv'))


if __name__ == '__main__':
    main()

