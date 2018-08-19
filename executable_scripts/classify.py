#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 11:21:56 2018

@author: miri-o
"""


import sys, argparse
import os
sys.path.insert(0, os.path.join(os.pardir, os.path.pardir))
from miris_tools.classifier import classifier
import pandas as pd
import numpy as np

def main(argv):
    
    parser=argparse.ArgumentParser(
            description='''classify.py function performs classification based on a feature table where each row is an observation and each column is a feature. It's output is a confusion matrix and a score.  ''',
            epilog="""All's well that ends well.""")
    parser = argparse.ArgumentParser()
    parser.add_argument('feature_file', help='a *.csv file containing the features table')
    parser.add_argument('--feature_cols', help='start and end range for features columns. default: ALL columns', nargs = 2, type = int)
    parser.add_argument('--labels_file', help='a file containing the labels for each row, default: None, using the features file')
    parser.add_argument('--labels_col_name', help='Name of the labels column, default: "labels"')
    parser.add_argument('-M','--model', help='Classification model. Currently supported: ["logistic_regression","decision_tree"] , default: "decision_tree"')
    
    args = parser.parse_args()
    if not os.path.isfile(args.feature_file) or args.feature_file[:-4] == '.csv':
        print('Feature file error! Make sure the file exists and it is *.csv file.\nExiting...')
        sys.exit(1)
    feature_file = pd.read_csv(args.feature_file)
    if not args.feature_cols:
        feat_range = (0, len(feature_file.columns))
    else:
        feat_range = (args.feature_cols[0], args.feature_cols[1]+1)
        features = feature_file.iloc[:, range(feat_range[0], feat_range[1])]
        
    if args.labels_file:
       if not os.path.isfile(args.feature_file) or args.feature_file[:-4] == '.csv':
           print('Labels file error! Make sure the file exists and it is *.csv file.\nExiting...')
           sys.exit(1)
       labels_file = pd.read_csv(args.labels_file)
    else:
       labels_file = feature_file.copy()
    if args.labels_col_name:
       if not args.labels_col_name in labels_file.columns:
           print('label column name doesnt exist.\nExiting...')
           sys.exit(1)
       else:
           labels_col_name = args.labels_col_name
    else: 
       labels_col_name = 'labels'
    labels = labels_file.loc[:, labels_col_name]
    if len(labels) != len(features):
           print('Labels and features length mismatch!.\nExiting...')
           sys.exit(1)
               
    if not args.model:
        model = 'decision_tree'
    else:
        model = args.model
        
        
    print("~~~~~~~~ Begin classifing...\nFeature_file: {}\nfeature columns: {}\nLabels column name: {}\nModel: {}\n~~~~~~~".format(
            os.path.abspath(args.feature_file), feat_range, labels_col_name, model))   
    
    our_classifier = classifier(features, labels, model)
    our_classifier.run()
    return(our_classifier.score)   
    
    

    
if __name__ == "__main__":
   main(sys.argv[1:])   
   
  