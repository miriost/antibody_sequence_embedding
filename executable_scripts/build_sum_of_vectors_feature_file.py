#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 13:25:45 2020

@author: miri-o
"""
import sys, argparse
import os
import pandas as pd


def main(argv):
    
    parser=argparse.ArgumentParser(
            description='''build_sum_of_vectors_feature_file.py function performs turns a large vector file to a feature file where each row is a sequence and each column is a dimension.''',
            epilog="""All's well that ends well.""")
    parser = argparse.ArgumentParser()
    parser.add_argument('--vector_file', help='a *.csv file containing the sequence vectors')
    
    parser.add_argument('--data_file', help='a file containing the labels for each row, default: None')
    parser.add_argument('--output_feature_file', help='output file for feature table')
    parser.add_argument('--subject_col_name', help='Name of the subject column, default: "SUBJECT"', default='SUBJECT')
    parser.add_argument('-M','--model', help='Classification model. Currently supported: "Nearest Neighbors (kNN)", "Linear SVM (LSVM)", "RBF SVM (RBF_SVM)", "Gaussian Process (Gaussian)", "Decision Tree (DT)", "Random Forest (RF)", "Neural Net (MLP)", "AdaBoost (Ada)", "Naive Bayes (NB)", "QDA" ,default: "decision_tree"', default = "decision_tree")
    
    args = parser.parse_args()
    
    if not (os.path.isfile(args.vector_file) and os.path.splitext(os.path.split(args.data_file)[1])[1]=='.csv'):
        print('Vector file {} error! Make sure the file exists and it is *.csv file.\nExiting...'.format(args.feature_file))
        sys.exit(1)
    vector_file = pd.read_csv(args.vector_file)
    #print(vector_file)
           
    if args.data_file:
       if not (os.path.isfile(args.data_file) and os.path.splitext(os.path.split(args.data_file)[1])[1]=='.csv'):
           print('Data file error! Make sure the file exists and it is *.csv file.\nExiting...')
           sys.exit(1)
       data_file = pd.read_csv(args.data_file)
       #print(data_file)
       if not args.subject_col_name in data_file.columns:
           print('subject column name doesnt exist.\nExiting...')
           sys.exit(1)
           
        #build a feature matrix, where each raw is a subject and each column is the dimension. the raw is composed of the sum of all the vectors of that subject.
        # add subject to vector_file
       subject_list = data_file.loc[:, args.subject_col_name]
       if len(subject_list) != len(vector_file):
           print('data and vectors length mismatch!.\nExiting...')
           sys.exit(1)       
       vector_file['subject'] = subject_list
       #print(subject_list)
       feature_table = vector_file.groupby('subject').sum()/vector_file.groupby('subject').count()
       print(feature_table)
       feature_table.to_csv(args.output_feature_file, index_label = 'subject')
       print('feature table saved to: {}'.format(args.output_feature_file))
       
       # Add label to each raw
       
        
#        
#       labels_file = feature_file.copy()
#       if not args.labels_col_name in data_file.columns:
#           print('label column name doesnt exist.\nExiting...')
#           sys.exit(1)
#       feature_file = feature_file.drop(args.labels_col_name, axis=1)
#    labels = labels_file.loc[:, args.labels_col_name]
#    
#    if args.labels_col_name:
#       if not args.labels_col_name in labels_file.columns:
#           print('label column name doesnt exist.\nExiting...')
#           sys.exit(1)
#       else:
#           labels_col_name = args.labels_col_name
#    else: 
#       labels_col_name = 'labels'
#    if not args.feature_cols:
#        feat_range = (1, len(feature_file.columns))
#    else:
#        feat_range = (args.feature_cols[0], args.feature_cols[1]+1)
#        
#    features = feature_file.iloc[:, range(feat_range[0], feat_range[1])]       
#
#    
#        
#        
#    print("~~~~~~~~ Begin classifing...\nFeature_file: {}\nfeature columns: {}\nLabels column name: {}\nModel: {}\n~~~~~~~".format(
#            os.path.abspath(args.feature_file), feat_range, labels_col_name, args.model))   
#    
#    
#    if args.model == 'regulized_logistic_regression' or args.model == 'RLR':
#        #plt.figure()
##        for i in range(1,11):
#        our_classifier = classifier(features, labels, args.model, C= .9)
#        our_classifier.run(n=1000)
#        print(our_classifier.score)
#        print(our_classifier.coef)
#        
##        for j in range(3):
#        plt.plot(range(args.feature_cols[0],args.feature_cols[1]+1), our_classifier.coef[0])      
#        plt.legend(loc = 'best')
#        
#        top_20_coefs = np.argsort(abs(our_classifier.coef[0]))[-20:]
#        plt.scatter(top_20_coefs, our_classifier.coef[0][top_20_coefs], c='r', marker='*')
#        print('Top 20 features:', str(top_20_coefs), '\nPerforming RLR...')
#        plt.show()
#        filtered_features = features.iloc[:, top_20_coefs]
#        our_classifier = classifier(filtered_features, labels, args.model, C= .9)
#        our_classifier.run(n=1000)
#        print(our_classifier.score)
#        
#        
#    else: 
#        our_classifier = classifier(features, labels, args.model)
#        our_classifier.run()
#        print(our_classifier.score)
#        
#
#

if __name__ == "__main__":
   main(sys.argv[1:])   
   