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
from random import shuffle
import numpy as np 
import pandas as pd

def main(argv):
    
    parser=argparse.ArgumentParser()
    parser.description=('''Split a data file to train and test sets based on labels''')
    parser.add_argument('data_file', 
                        help='a file raws of data, including labaled')
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
        
    
    # get list of subjects and shuffle them
    subjects = shuffle(np.unique(data_file[args.labels_col_name]))
    N = len(subjects) #Number of subject
    print('Total number of subjects: ' , str(N))
    folds = args.f
    
    
#        if len(data_files) % folds != 0:
#        raise ValueError(
#            "invalid number of folds ({}) for the number of "
#            "documents ({})".format(folds, len(data_files))
#        )
    fold_size = N // folds
    for split_index in range(0, N, fold_size):
        testing = subjects[split_index:split_index + fold_size]
        training = subjects[:split_index] + subjects[split_index + fold_size:]
        print('Testing indices [{}:{}], training indices: [{}:{}, {}:]'.format(split_index, split_index + fold_size, 0, split_index, split_index + fold_size))
        yield training, testing
        
    
if __name__ == "__main__":
   main(sys.argv[1:])   
   