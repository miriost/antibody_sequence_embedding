#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 11:21:56 2018

@author: miri-o
"""
import sys, argparse
import os
import matplotlib.pyplot as plt
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()).split('antibody_sequence_embedding')[0]) 
from antibody_sequence_embedding.classifier import classifier
import pandas as pd
import numpy as np


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', "True"):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', "False"):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main(argv):
    
    parser=argparse.ArgumentParser(
            description='''classify.py function performs classification based on a feature table where each row is an 
            observation and each column is a feature. It's output is a confusion matrix and a score.  ''',
            epilog="""All's well that ends well.""")
    parser = argparse.ArgumentParser()
    parser.add_argument('feature_file', help='a *.csv file containing the features table')
    parser.add_argument('--feature_cols', help='start and end range for features columns. default: ALL columns',
                        nargs = 2, type = int)
    parser.add_argument('--labels_file', help='a file containing the labels for each row, default: None, '
                                              'using the features file')
    parser.add_argument('--labels_col_name', help='Name of the labels column, default: "labels"', default='labels')
    parser.add_argument('-M','--model', help='Classification model. current supported models: logistic_regression, '
                                             'decision_tree, kNN, linear_svm, RBF_SVM, Gaussian, Random_Forest, MLP,'
                                             ' ADA, MLP, naive_bayes, QDA, all - to run through all models" , '
                                             'default: "decision_tree"', default = "decision_tree")
    parser.add_argument('-V', '--verify', type=str2bool, default=False, help= 'Verify results on un-seen data set')
    parser.add_argument('--verify_feature_file', help = 'verification feature file')
    
    
    args = parser.parse_args()
    
    if not (os.path.isfile(args.feature_file) and os.path.splitext(os.path.split(args.feature_file)[1])[1] == '.csv'):
        print('Feature file {} error! Make sure the file exists and it is *.csv file.\nExiting...'.format(args.feature_file))
        sys.exit(1)
    feature_file = pd.read_csv(args.feature_file)
           
    if args.labels_file:
        if not (os.path.isfile(args.feature_file) and os.path.splitext(os.path.split(args.feature_file)[1])[1] == '.csv'):
            print('Labels file error! Make sure the file exists and it is *.csv file.\nExiting...')
            sys.exit(1)
        labels_file = pd.read_csv(args.labels_file)
        if args.labels_col_name not in labels_file.columns:
            print('label column name doesnt exist.\nExiting...')
            sys.exit(1)
    else:
        labels_file = feature_file.copy()
        if args.labels_col_name not in labels_file.columns:
            print('label column name doesnt exist.\nExiting...')
            sys.exit(1)
        feature_file = feature_file.drop(args.labels_col_name, axis=1)
    labels = labels_file.loc[:, args.labels_col_name]
    
    if args.labels_col_name:
        if args.labels_col_name not in labels_file.columns:
            print('label column name doesnt exist.\nExiting...')
            sys.exit(1)
        else:
            labels_col_name = args.labels_col_name
    else: 
        labels_col_name = 'labels'
    if not args.feature_cols:
        feat_range = (1, len(feature_file.columns))
    else:
        feat_range = (args.feature_cols[0], args.feature_cols[1]+1)
    print('features range: {},{}'.format(feat_range[0], feat_range[1]))
    print('features used:{}'.format(feature_file.columns[feat_range[0]: feat_range[1]]))
    features = feature_file.iloc[:, range(feat_range[0], feat_range[1])]       
    if len(labels) != len(features):
        print('Labels and features length mismatch!.\nExiting...')
        sys.exit(1)
    if args.verify:
        if not(os.path.isfile(args.verify_feature_file)):
            print('verify feature table file error, make sure feature file path\nExiting...')
            sys.exit(1)
        else:
            verify_feature_file = pd.read_csv(args.verify_feature_file, index_col = 0)
            X_verify = verify_feature_file.drop(args.labels_col_name, axis=1)
            y_verify = verify_feature_file.loc[:, args.labels_col_name]
        
    if args.model == 'all' or args.model=='All':
        model_list = ['logistic_regression', "decision_tree", "kNN", "linear_svm", "RBF_SVM", "Gaussian",
                      "Random_Forest", "MLP", "ADA", "MLP", "naive_bayes", "QDA"]
    else:
        model_list = [args.model]

    for model in model_list:
        print("~~~~~~~~ Begin classifing...\nFeature_file: {}\nfeature columns: {}\nLabels column name: {}\nModel: {}\n~~~~~~~".format(
                os.path.abspath(args.feature_file), feat_range, labels_col_name, model))

        if model == 'regulized_logistic_regression' or model == 'RLR':
            #plt.figure()
    #        for i in range(1,11):
            our_classifier = classifier(features, labels, model, C= .9)
            our_classifier.run(n=1000)
            print(our_classifier.score)
            print(our_classifier.coef)

    #        for j in range(3):
            plt.plot(range(args.feature_cols[0],args.feature_cols[1]+1), our_classifier.coef[0])
            plt.legend(loc = 'best')

            top_20_coefs = np.argsort(abs(our_classifier.coef[0]))[-20:]
            plt.scatter(top_20_coefs, our_classifier.coef[0][top_20_coefs], c='r', marker='*')
            print('Top 20 features:', str(top_20_coefs), '\nPerforming RLR...')
            plt.show()
            filtered_features = features.iloc[:, top_20_coefs]
            our_classifier = classifier(filtered_features, labels, model, C= .9)
            our_classifier.run(n=1000)
            print(our_classifier.score)

        else:
            our_classifier = classifier(features, labels, model)
            if not args.verify:
                print('Performing classification, cross validation without verification')
                our_classifier.run()
                print(our_classifier.score)  #score on test set
            else:
                print('Performing classification, cross validation with verification')
                our_classifier.run(validate=True, X_validate=X_verify, y_validate=y_verify)
                print('Scores on test set:')
                print(our_classifier.score)  #score on test set
                print('Scores on verification set:')
                print(our_classifier.validation_score)  #score on test set

if __name__ == "__main__":
    main(sys.argv[1:])
   
   
  
