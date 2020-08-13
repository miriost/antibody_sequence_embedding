# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 10:12:56 2020

@author: mirio
"""

# split a data file to train and test data sets
# input - data file
# output - 2 files, test and train


import sys, argparse, os
sys.path.insert(0, "/media/miri-o/Documents")
from random import shuffle, seed
import numpy as np 
import pandas as pd
from datetime import datetime


def main():
    seed(0)
    parser=argparse.ArgumentParser()
    parser.description=('''Split a data file to train and test data sets based on labels, according to the number of folds''')
    parser.add_argument('data_file', 
                        help='a file containing raws of data, including labals')
    parser.add_argument('vectors_file', 
                        help='a file with the same length as data, each raw is the junction vector')
    parser.add_argument('-l', '--labels_col_name', 
                        help='Name of the labels column in data file, default: "labels"', default= 'FILENAME')
    parser.add_argument('-f', '--number_of_folds', 
                        help='the number of folds, influences the test size [number of subjects/number of folds], default: "5"', default= 5)

    args = parser.parse_args()
    
    if not os.path.isfile(args.data_file) or args.data_file[:-4] == '.csv':
        print('feature file error! Make sure the file exists and it is *.csv file.\nExiting...')
        sys.exit(1)
    data_file = pd.read_csv(args.data_file)
    print('Data file loaded, originial file length: ', str(len(data_file)))
    
    if not args.labels_col_name in data_file.columns:
        print(data_file.columns)
        print(args.labels_col_name + ' column is missing in data file\nExiting...')
        sys.exit(1)
        
    if not os.path.isfile(args.vectors_file) or args.vectors_file[:-4] == '.csv':
        print('vectors file error! Make sure the file exists and it is *.csv file.\nExiting...')
        sys.exit(1)
    vectors_file = pd.read_csv(args.vectors_file)
    print('vectors_file file loaded, originial file length: ', str(len(vectors_file)))
    
    if len(vectors_file) != len(data_file):
        print('vectors file and data file length mismatch!\nExiting...')
        sys.exit(1)
    
    # get list of subjects and shuffle them
    subjects = np.unique(data_file[args.labels_col_name])
    #print('Subjects before shuffle:', subjects)
    shuffle(subjects)
    #print('Subjects after shuffle:', subjects)
    N = len(subjects) #Number of subjects
    print('Total number of subjects: ' , str(N))
    folds = args.number_of_folds
    
    fold_size = N//folds
    fold_id = 0
    
    by_subject = data_file.groupby([args.labels_col_name])
    
    for split_index in range(0, N-fold_size, fold_size):
        testing = subjects[(split_index):(split_index + fold_size)]
        #print(testing)
        training = np.concatenate((subjects[:(split_index)],subjects[(split_index + fold_size):]), axis = None)
        #print(training)
        print('Testing indices [{}:{}], training indices: [{}:{}, {}:]'.format(split_index, split_index + fold_size, 0, split_index, split_index + fold_size))


        sub_train = 0
        sub_test = 0
        sub_num = 0
    
        output_train_data_df = pd.DataFrame(columns = list(data_file.columns))
        output_train_vectors_df = pd.DataFrame(columns = list(vectors_file.columns))
        output_test_data_df = pd.DataFrame(columns = list(data_file.columns))
        output_test_vectors_df = pd.DataFrame(columns = list(vectors_file.columns))
    
        
        for subject, frame in by_subject:
            sub_num +=1
            print("------------------------")
            print(f"Analysing {subject!r} {sub_num!r}")
            dateTimeObj = datetime.now()
            timeObj = dateTimeObj.time()
            if subject in testing:
                sub_test +=1
                output_test_data_df = output_test_data_df.append(data_file.iloc[frame.index])
                output_test_vectors_df = output_test_vectors_df.append(vectors_file.iloc[frame.index])
                print(f"subject {subject!r} added to test set")
            elif subject in training:
                sub_train +=1
                output_train_data_df = output_train_data_df.append(data_file.iloc[frame.index])
                output_train_vectors_df = output_train_vectors_df.append(vectors_file.iloc[frame.index])
                print(f"subject {subject!r} added to train set")
            else:
                print(f'Error, subject {subject!r} not found in train or test lists')
               
                
            
        print(f'----- Summary fold {fold_id} -----')
        print(f'Analysed {sub_num!r} subjects')
        print(f'{sub_train} subjects to train set, {sub_test} subjects to test set')
        
        output_train_data_df.to_csv(os.path.splitext(args.data_file)[0]+'_TRAIN_' + str(fold_id) + os.path.splitext(args.data_file)[1], index = False)
            
        output_test_data_df.to_csv(os.path.splitext(args.data_file)[0]+'_TEST_' + str(fold_id) + os.path.splitext(args.data_file)[1], index = False)
        
        output_train_vectors_df.to_csv(os.path.splitext(args.vectors_file)[0]+'_TRAIN_' + str(fold_id) + os.path.splitext(args.vectors_file)[1], index = False)
        
        output_test_vectors_df.to_csv(os.path.splitext(args.vectors_file)[0]+'_TEST_' + str(fold_id) + os.path.splitext(args.vectors_file)[1], index = False)
        print('All files saved')
        fold_id+=1
    


if __name__ == "__main__":
   main()   
   
