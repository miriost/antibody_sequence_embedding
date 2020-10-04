# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 10:12:56 2020

@author: mirio
"""

# split a data file to train and test data sets
# input - data file and optional vector file
# output - 2 files, test and train

import sys, argparse, os
sys.path.insert(0, "/media/miri-o/Documents")
from random import shuffle, seed, sample
import numpy as np
import pandas as pd
import math


def main():
    seed(0)
    parser = argparse.ArgumentParser()
    parser.description = 'Split a data file to train and test data sets based on labels, according to the number of ' \
                         'folds'
    parser.add_argument('data_file', 
                        help='a file containing raws of data, including labals')
    parser.add_argument('-r', '--repeated',
                        help='use repeated sampling for the folds. default=False', type=bool, default=False)
    parser.add_argument('--test_size',
                        help='Test set size as percentage from data. If repeated is False this argument is ignored.'
                             ' Default=20', type=int, default=20)
    parser.add_argument('-v', '--vectors_file',
                        help='a file with the same length as data, each raw is the junction vector')
    parser.add_argument('-l', '--id_col_name',
                        help='Name of the subjects id column in data file, default: "FILENAME"', type=str,
                        default= 'FILENAME')
    parser.add_argument('-l', '--labels_col_name', 
                        help='Name of the labels column in data file, default: "labels"', type=str, default= 'labels')
    parser.add_argument('-f', '--number_of_folds', 
                        help='the number of folds, influences the test size [number of subjects/number of folds], '
                             'default: "5"', default=5)

    args = parser.parse_args()
    
    if not os.path.isfile(args.data_file) or args.data_file[:-4] == '.csv':
        print('feature file error! Make sure the file exists and it is *.csv file.\nExiting...')
        sys.exit(1)
    data_file = pd.read_csv(args.data_file, sep='\t')
    print('Data file loaded, originial file length: ', str(len(data_file)))
    
    if args.labels_col_name is not in data_file.columns:
        print(data_file.columns)
        print(args.labels_col_name + ' column is missing in data file\nExiting...')
        sys.exit(1)

    if args.id_col_name is not in data_file.columns:
        print(data_file.columns)
        print(args.id_col_name + ' column is missing in data file\nExiting...')
        sys.exit(1)
        
    if not os.path.isfile(args.vectors_file) or args.vectors_file[:-4] == '.csv':
        print('vectors file error! Make sure the file exists and it is *.csv file.\nExiting...')
        sys.exit(1)

    vectors_file = None
    if args.vectors_file is not None:
        vectors_file = pd.read_csv(args.vectors_file)
        print('vectors_file file loaded, original file length: ', str(len(vectors_file)))
    
    if len(vectors_file) != len(data_file):
        print('vectors file and data file length mismatch!\nExiting...')
        sys.exit(1)

    if args.repeated and (args.test_size < 1 or args.test_size > 50):
        print('test_size argument is given in percentage, value should be in the range [1,50]\nExiting...')
        sys.exit(1)

    subjects_df = pd.DataFrame(np.unique(data_file[args.id_col_name]), columns=['subject'])
    subjects_df['labels'] = [data_file.loc[data_file[args.id_col_name] == subject, args.label_col_name].iloc[0]
                             for subject in subjects_df.loc[:, 'subject']]
    n_subjects = len(subjects_df) #Number of subjects
    print('Total number of subjects: ' , str(n_subjects))
    n_folds = args.number_of_folds

    by_subject = data_file.groupby([args.id_col_name])
    if not args.repeated:
        output_folds = create_folds_no_repetitions(subjects_df, n_folds)
    else:
        test_size = (args.test_size * n_subjects) // 100
        if round(test_size * len(subjects_df)) < len(np.unique(subjects_df.loc[:,'labels'])):
            print('test_size is too small\nExiting...')
            sys.exit(1)
        output_folds = create_folds_with_repetitions(subjects_df, n_folds, test_size)

    output_train_data_df = pd.DataFrame(columns=list(data_file.columns))
    output_test_data_df = pd.DataFrame(columns=list(data_file.columns))
    output_test_vectors_df = None
    output_train_vectors_df = None
    if vectors_file is not None:
        output_train_vectors_df = pd.DataFrame(columns=list(vectors_file.columns))
        output_test_vectors_df = pd.DataFrame(columns = list(vectors_file.columns))

    fold_id = 0
    for (training, testing) in output_folds:
        sub_test = 0
        sub_train = 0
        for subject, frame in by_subject:
            if subject in testing:
                sub_test += 1
                output_test_data_df = output_test_data_df.append(data_file.iloc[frame.index])
                if vectors_file is not None:
                    output_test_vectors_df = output_test_vectors_df.append(vectors_file.iloc[frame.index])
                print(f"subject {subject!r} added to test set")
            elif subject in training:
                sub_train += 1
                output_train_data_df = output_train_data_df.append(data_file.iloc[frame.index])
                if vectors_file is not None:
                    output_train_vectors_df = output_train_vectors_df.append(vectors_file.iloc[frame.index])
                print(f"subject {subject!r} added to train set")
            else:
                print(f'Error, subject {subject!r} not found in train or test lists')

        print(f'----- Summary fold {fold_id} -----')
        print(f'{sub_train} subjects to train set, {sub_test} subjects to test set')

        output_train_data_df.to_csv(os.path.splitext(args.data_file)[0] + '_TRAIN_' + str(fold_id) +
                                    os.path.splitext(args.data_file)[1], sep='\t', index=False)
        output_test_data_df.to_csv(os.path.splitext(args.data_file)[0] + '_TEST_' + str(fold_id) +
                                   os.path.splitext(args.data_file)[1], sep='\t', index=False)

        if vectors_file is not None:
            output_train_vectors_df.to_csv(os.path.
                                           (args.vectors_file)[0] + '_TRAIN_' + str(fold_id) +
                                           os.path.splitext(args.vectors_file)[1], index=False)
            output_test_vectors_df.to_csv(os.path.splitext(args.vectors_file)[0] + '_TEST_' + str(fold_id) +
                                          os.path.splitext(args.vectors_file)[1], index=False)

        print(f'----- fold {fold_id} files saved -----')
        fold_id += 1


def create_folds_with_repetitions(subjects_df, n_folds, test_size):

    n_test_subjects = round(len(subjects_df) * test_size)
    labels = np.unique(subjects_df['labels'])
    subjects = np.array(subjects_df.loc[:, 'subject'])

    output_folds = []
    for fold in range(n_folds):
        testing = np.array([])
        for label in labels:
            label_subjects = subjects_df.loc[subjects_df['labels'] == label, 'subject']
            # try to create test set as balanced as possible, an equal number of subjects
            # from each label, if the number of subjects with this label allows it
            n_label = min(n_test_subjects // len(labels), math.ceil(len(label_subjects) * test_size))
            testing = np.append(testing, sample(label_subjects.to_list(), n_label))

        training = subjects[np.in1d(subjects, testing) == False]
        if len(testing) < n_test_subjects:
            testing = np.append(testing, training.sample(n_test_subjects-len(testing)))
            training = subjects[np.in1d(subjects, testing) == False]

        output_folds += [(training, testing)]

    return output_folds


def test_create_folds_with_repetitions():

    subjects_df = pd.DataFrame()
    subjects_df['subject'] = ['P1_I1', 'P1_I2', 'P1_I3', 'P1_I4', 'P1_I5', 'P1_I6', 'P1_I7', 'P1_I8', 'P1_I9', 'P1_I10',
                              'P1_I11', 'P1_I12', 'P1_I13', 'P1_I14', 'P1_I15', 'P1_I16', 'P1_I17', 'P1_I18', 'P1_I19']
    subjects_df['labels'] = 0
    subjects_df.loc[sample(list(range(len(subjects_df))), 7) , 'labels'] = 1

    n_folds = 4
    test_size = 0.2
    output_folds = create_folds_with_repetitions(subjects_df, n_folds, test_size)
    return output_folds


def create_folds_no_repetitions(subjects_df, n_folds):

    fold_size = len(subjects_df) // n_folds

    subjects = np.array(subjects_df.loc[:, 'subject'])
    shuffle(subjects)
    output_folds = []
    for split_index in range(0, len(subjects)-fold_size, fold_size):
        testing = subjects[split_index:(split_index + fold_size)]
        training = np.concatenate((subjects[:split_index],subjects[(split_index + fold_size):]), axis = None)
        print('Testing indices [{}:{}], training indices: [{}:{}, {}:]'.format(split_index, split_index + fold_size, 0,
                                                                               split_index, split_index + fold_size))
        output_folds += [(training, testing)]

    return output_folds


def test_create_folds_no_repetitions():

    subjects_df = pd.DataFrame()
    subjects_df['subject'] = ['P1_I1', 'P1_I2', 'P1_I3', 'P1_I4', 'P1_I5', 'P1_I6', 'P1_I7', 'P1_I8', 'P1_I9', 'P1_I10',
                              'P1_I11', 'P1_I12', 'P1_I13', 'P1_I14', 'P1_I15', 'P1_I16', 'P1_I17', 'P1_I18', 'P1_I19']
    subjects_df['labels'] = 0
    subjects_df.loc[sample(list(range(len(subjects_df))), 7) , 'labels'] = 1

    n_folds = 4
    output_folds = create_folds_no_repetitions(subjects_df, n_folds)
    return output_folds


if __name__ == "__main__":
   main()
