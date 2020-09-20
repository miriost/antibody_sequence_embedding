#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 15:31:14 2018

@author: miri-o
"""

import pandas as pd
import numpy as np
import sys
import os
import random

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
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure



def main(filtered_file, vector_path, sep):
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
    data = pd.read_csv(filtered_file, sep = sep)
  
    valid_indexes = data.index[~data.V_FAMILY.isnull()] # Remove rows where v_family is null 
    np.random.seed(1)
    sub_indexes = [*np.random.choice(data.index[data.V_FAMILY=='IGHV1'], 114000, replace=False), *np.random.choice(data.index[data.V_FAMILY=='IGHV3'], 114000, replace=False), *np.random.choice(data.index[data.V_FAMILY=='IGHV4'], 114000, replace=False)]



    dimensions = [2, 5, 10, 15, 20, 30, 50, 75, 90, 100]
    decision_tree_f1_score = []
    knn_f1_score = []

    name = filtered_file.split('FILTERED')[0]

    for dim in dimensions:
        vector_file = vector_path + name + str(dim) + 'D_PCA.csv'
        if not os.path.isfile(vector_file):
            continue;
        vectors = pd.read_csv(vector_file, sep = ',')
        print('--------------\n' + str(dim) + ' Dimensions ~~~\n')  
        if len(vectors) == len(data):
            print('Data length validation succeeded')
        else:
            print('Data validation FAILED...')
            sys.exit(1)
        #X = vectors.loc[valid_indexes]
        #y = data.V_FAMILY.loc[valid_indexes]
    
        #sampled data
        X = vectors.loc[sub_indexes]
        y = data.V_FAMILY.loc[sub_indexes]
    
        y = [int(x[-1]) for x in y]
        
        # Create Training and Test Sets and Apply Scaling
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    #    X_train = scaler.fit_transform(X_train)
    #    X_test = scaler.transform(X_test)
    
        #Logistic regression
        
    #    logreg = LogisticRegression()
    #    logreg.fit(X_train, y_train)
    #    print('Accuracy of Logistic regression classifier on training set: {:.2f}'
    #         .format(logreg.score(X_train, y_train)))
    #    print('Accuracy of Logistic regression classifier on test set: {:.2f}'
    #         .format(logreg.score(X_test, y_test)))
    #    
        #Decision Tree
    
        clf = DecisionTreeClassifier().fit(X_train, y_train)
        print('Accuracy of Decision Tree classifier on training set: {:.2f}'
            .format(clf.score(X_train, y_train)))
        print('Accuracy of Decision Tree classifier on test set: {:.2f}'
            .format(clf.score(X_test, y_test)))
        decision_tree_f1_score.append(clf.score(X_test, y_test))
    
        pred = clf.predict(X_test)
        print(confusion_matrix(y_test, pred))
        print(classification_report(y_test, pred))
        #K-Nearest Neighbors
    
        knn = KNeighborsClassifier()
        knn.fit(X_train, y_train)
        print('Accuracy of K-NN classifier on training set: {:.2f}'
            .format(knn.score(X_train, y_train)))
        print('Accuracy of K-NN classifier on test set: {:.2f}'
            .format(knn.score(X_test, y_test)))
        knn_f1_score.append(knn.score(X_test, y_test))
        pred = knn.predict(X_test)
        print(confusion_matrix(y_test, pred))
        print(classification_report(y_test, pred))
    
        #Linear Discriminant Analysis
        
    #    lda = LinearDiscriminantAnalysis()
    #    lda.fit(X_train, y_train)
    #    print('Accuracy of LDA classifier on training set: {:.2f}'
    #         .format(lda.score(X_train, y_train)))
    #    print('Accuracy of LDA classifier on test set: {:.2f}'
    #         .format(lda.score(X_test, y_test)))
    #    
    #    #Gaussian Naive Bayes
    #    
    #    gnb = GaussianNB()
    #    gnb.fit(X_train, y_train)
    #    print('Accuracy of GNB classifier on training set: {:.2f}'
    #         .format(gnb.score(X_train, y_train)))
    #    print('Accuracy of GNB classifier on test set: {:.2f}'
    #         .format(gnb.score(X_test, y_test)))
    #    
    #    ## Support Vector Machine
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
    print(f'Dimensions: {dimensions}')
    print(f'k-nn f1 score: {knn_f1_score}')
    print(f'Decision tree f1 score: {decision_tree_f1_score}')
  
    knn_f1_score.insert(0,0)
    decision_tree_f1_score.insert(0,0)
    
    figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(dimensions, decision_tree_f1_score, '*-', )
    plt.ylim(0,1)
    plt.xlim(0,100)
    plt.plot(dimensions, knn_f1_score, '*-', c = 'r')
    plt.legend(['Decision tree', 'k-NN'])
    plt.title('F1-score of V family classifiers', fontsize = 'x-large')
    plt.xlabel('Number of dimensions')
    plt.ylabel('F1-score')
    plt.show()
    
    #https://towardsdatascience.com/solving-a-simple-classification-problem-with-python-fruits-lovers-edition-d20ab6b071d2


if __name__ == '__main__':
    if len(sys.argv) < 4:
        sys.exit('usage: python V_family_analysis_different dimensions.py <filtered csv file> <vectores directory> <sep>')
    main(sys.argv[1], sys.argv[2], sys.argv[3])
