#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 10:34:20 2018

@author: miri-o
"""

import sys, argparse
import os
sys.path.insert(0, os.path.join(os.pardir, os.path.pardir))
import pandas as pd


def main(argv):
    
    parser=argparse.ArgumentParser(description='''Add labels to feature table''')
    parser.add_argument('feature_file', 
                        help='a *.csv file containing the features table')
    parser.add_argument('labels_file', 
                        help='a file containing the labels for each observation')
    parser.add_argument('-l', '--labels_col_name', 
                        help='Name of the labels column in label file, default: "labels"', default= 'labels')
    parser.add_argument('-o', '--observation_col_name', 
                        help='Name of the observation column in labels file, default: "SUBJECT"', default = "SUBJECT")
    parser.add_argument('-od', '--observation_col_name_data_file', 
                        help='Name of the observation column in features file, default: "SUBJECT"', default = "SUBJECT")
    
    args = parser.parse_args()
    print('~~~\nAdding labels to feature file, feature file: {}, labels file: {}, labels column name: {}, observation column name: {}'.format(args.feature_file,
          args.labels_file, args.labels_col_name, args.observation_col_name))
    
    if not os.path.isfile(args.feature_file) or args.feature_file[:-4] == '.csv':
           print('feature file error! Make sure the file exists and it is *.csv file.\nExiting...')
           sys.exit(1)

    if not os.path.isfile(args.labels_file) or args.labels_file[:-4] == '.csv':
           print('labels file error! Make sure the file exists and it is *.csv file.\nExiting...')
           sys.exit(1)
    
    
    feature_file = pd.read_csv(args.feature_file, sep='\t')     ## removed index_col=0!!!
    labels_file = pd.read_csv(args.labels_file) 
    if not args.observation_col_name in labels_file.columns:
        print(labels_file.columns)
        print(args.observation_col_name + ' column is missing in labels file\nExiting...')
        sys.exit(1)
    if not args.labels_col_name in labels_file.columns:
        print(labels_file.columns)
        print(args.labels_col_name + ' column is missing in labels file\nExiting...')
        sys.exit(1)           
    if not args.observation_col_name_data_file in feature_file.columns:
        print(feature_file.columns)
        print(args.observation_col_name_data_file + ' column is missing in feature file\nExiting...')
        sys.exit(1)  
#    labels_file = labels_file[[args.observation_col_name, args.labels_col_name]]
#    print(labels_file.head())
#    labels_dict = labels_file.to_dict()
    labels_dict = {}
    for obs, lab in zip(labels_file[args.observation_col_name], labels_file[args.labels_col_name]):
        labels_dict[obs] = lab
        

#    labels = [labels_dict[args.labels_col_name][i] for i in feature_file.index]
    labels = [labels_dict[i] for i in feature_file.loc[:, args.observation_col_name_data_file]]
    feature_file['labels'] = labels
    
    path = os.path.join(os.path.dirname(args.feature_file),
                        os.path.split(args.feature_file)[-1].split('.tab')[0] + '_with_labels.tab')
    feature_file.to_csv(path, index=False, sep='\t')
    print('~~~\nlabels add to feature file, new file is: ' + path)

      
if __name__ == "__main__":
   main(sys.argv[1:])   
   
  
