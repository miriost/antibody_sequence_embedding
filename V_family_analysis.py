#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 15:31:14 2018

@author: miri-o
"""

import pandas as pd
import numpy as np
import sys, argparse
import os
sys.path.insert(0, os.path.join(os.pardir, os.path.pardir))
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

## Preprocess
#file = pd.read_csv('/media/miri-o/Documents/filtered_data_sets/Celiac_for_V_family_analysis_3M_seqs_celiac_model_Jan19_2019_FILTERED_DATA.csv',sep=',')
#file_with_labels = pd.read_csv('/media/miri-o/Documents/filtered_data_sets/Celiac_for_V_Family_V_family_labels_Jan9_2019.csv', index_col = 0, sep = '\t')
#valid_labels = ['NA' if ',' in y else y for y in file_with_labels.V_FAMILY]
#file_with_labels['V_FAMILY'] = valid_labels
#file_with_labels.to_csv('/media/miri-o/Documents/filtered_data_sets/Celiac_for_V_family_analysis_3M_seqs_celiac_model_Jan19_2019_FILTERED_DATA2.csv', sep = ',')

# Now let's try straight-forawrd classification on our vectors based on the v family
#vectors = pd.read_csv('/media/miri-o/Documents/vectors/Celiac_for_V_family_analysis_3M_seqs_Celiac_n_3_trimming_2_1_VECTORS_after_PCACeliac_3D.csv')
#vectors = pd.read_csv(r'C:\Users\mirio\Dropbox\BIU\LAB\Celiac_for_V_family_analysis_10K_3D_vectors.csv')
#data = pd.read_csv(r'C:\Users\mirio\Dropbox\BIU\LAB\Celiac_for_V_family_analysis_10K_DATA.csv')

# Trying on all data (3M)


def run_model(model, model_name, X_train, y_train, X_test, y_test):
    dateTimeObj = datetime.now()
    timeObj = dateTimeObj.time()
    print(f'{timeObj} Beginning data analysis on {model_name} model')
    model.fit(X_train, y_train)
    print('Accuracy of {} classifier on training set: {:.2f}'.format(model_name, model.score(X_train, y_train)))
    print('Accuracy of {} classifier on test set: {:.2f}'.format(model_name, model.score(X_test, y_test)))  
    pred = model.predict(X_test)
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred))

def main(argv):
    
    parser=argparse.ArgumentParser(description='''Classify V family''')
    parser.add_argument('vectors_file', 
                        help='a *.csv file containing the vectors')
    parser.add_argument('data_file', 
                        help='a file containing the labels for each observation')
    parser.add_argument('-l', '--labels_col_name', 
                        help='Name of the labels column in data file, default: "labels"', default= 'V_FAMILY')
    parser.add_argument('-len', '--sample_length', 
                        help='number of samples from each group [K], default: "100"', default= 100)

    args = parser.parse_args()
    print('~~~\nClassify v family, vactor file: {}, data file: {}, labels column name: {}, sample length: {}K'.format(args.vectors_file,
          args.data_file, args.labels_col_name, args.sample_length))
    
    if not os.path.isfile(args.vectors_file) or args.vectors_file[:-4] == '.csv':
           print('vectors file error! Make sure the file exists and it is *.csv file.\nExiting...')
           sys.exit(1)

    if not os.path.isfile(args.data_file) or args.data_file[:-4] == '.csv':
           print('data file error! Make sure the file exists and it is *.csv file.\nExiting...')
           sys.exit(1)
    
    
    vectors = pd.read_csv(args.vectors_file, sep = ',')     ## removed index_col=0!!!
    data = pd.read_csv(args.data_file, sep = '\t') 
    labels_col_name = str(args.labels_col_name)
    
    if len(vectors) == len(data):
        print('Data length validation succeeded')
    else:
        print('Data validation FAILED...')
        sys.exit(1)
    if not args.labels_col_name in data.columns:
        print(data.columns)
        print(args.labels_col_name + ' column is missing in labels file\nExiting...')
        sys.exit(1)
        
    print(len(data))    
    valid_indexes = data.index[~data[labels_col_name].isnull()] # Remove rows where v_family is null 

    print(len(data))
    sub_indexes = [*np.random.choice(data.index[data[labels_col_name]=='IGHV1'], int(args.sample_length)*1000, replace=False), *np.random.choice(data.index[data[labels_col_name]=='IGHV3'], int(args.sample_length)*1000, replace=False), *np.random.choice(data.index[data[labels_col_name]=='IGHV4'], int(args.sample_length)*1000, replace=False)]
    #X = vectors.loc[valid_indexes]
    #y = data.V_FAMILY.loc[valid_indexes]
    
    #sampled data
    X = vectors.loc[sub_indexes]
    y = data[labels_col_name].loc[sub_indexes]
    
    y = [int(x[-1]) for x in y]
    
    # Create Training and Test Sets and Apply Scaling
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    #X_train = scaler.fit_transform(X_train)
    #X_test = scaler.transform(X_test)
    
    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM",
         "Decision Tree", "Random Forest", "AdaBoost",
         "Naive Bayes"]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
#        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
#        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB()]
#        QuadraticDiscriminantAnalysis()]

 # iterate over classifiers
    for name, clf in zip(names, classifiers):
        run_model(clf, name, X_train, y_train, X_test, y_test)

#        
#    #Logistic regression
#    model = LogisticRegression()
#    run_model(model, X_train, y_train, X_test, y_test)
#    #Decision Tree
#    model = DecisionTreeClassifier()
#    run_model(model, X_train, y_train, X_test, y_test)
#    
#    #K-Nearest Neighbors
#    model = KNeighborsClassifier()
#    run_model(model, X_train, y_train, X_test, y_test)
#    
#    #Gaussian Naive Bayes
#    model = GaussianNB()
#    run_model(model, X_train, y_train, X_test, y_test)
#    
#    #ADAboost
#    model = AdaBoostClassifier()
#    run_model(model, X_train, y_train, X_test, y_test)
    
if __name__ == "__main__":
   main(sys.argv[1:])   
   
  







### Support Vector Machine
#def svm(X_train, y_train, X_test, y_test):
#    svm = SVC()
#    svm.fit(X_train, y_train)
#    print('Accuracy of SVM classifier on training set: {:.2f}'
#         .format(svm.score(X_train, y_train)))
#    print('Accuracy of SVM classifier on test set: {:.2f}'
#         .format(svm.score(X_test, y_test)))
#    
#    pred = svm.predict(X_test)
#    print(confusion_matrix(y_test, pred))
#    print(classification_report(y_test, pred))



#https://towardsdatascience.com/solving-a-simple-classification-problem-with-python-fruits-lovers-edition-d20ab6b071d2