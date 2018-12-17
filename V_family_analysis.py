#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 15:31:14 2018

@author: miri-o
"""

import pandas as pd
import numpy as np
import sys

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
## Preprocess
#file = pd.read_csv('/media/miri-o/Documents/filtered_data_sets/Celiac_for_V_family_analysis_3M_seqs_Celiac_n_3_trimming_2_1_FILTERED_DATA.csv',sep=',')
#labels = pd.read_csv('/media/miri-o/Documents/filtered_data_sets/Celiac_for_V_Family_V_family_labels.csv', sep='\t')
#valid_labels = ['NA' if ',' in y else y for y in labels.x ]
#file['V_FAMILY'] = valid_labels
#file.to_csv('/media/miri-o/Documents/filtered_data_sets/Celiac_for_V_family_analysis_3M_seqs_Celiac_n_3_trimming_2_1_labeled_FILTERED_DATA.csv', sep = ',')

# Now let's try straight-forawrd classification on our vectors based on the v family
#vectors = pd.read_csv('/media/miri-o/Documents/vectors/Celiac_for_V_family_analysis_3M_seqs_Celiac_n_3_trimming_2_1_VECTORS_after_PCACeliac_3D.csv')
#vectors = pd.read_csv(r'C:\Users\mirio\Dropbox\BIU\LAB\Celiac_for_V_family_analysis_10K_3D_vectors.csv')
#data = pd.read_csv(r'C:\Users\mirio\Dropbox\BIU\LAB\Celiac_for_V_family_analysis_10K_DATA.csv')

# Trying on all data (3M)

vectors = pd.read_csv('/media/miri-o/Documents/vectors/Celiac_for_V_family_analysis_3M_seqs_Celiac_n_3_trimming_2_1_VECTORS_after_PCACeliac_10D.csv', sep = ',')
data = pd.read_csv('/media/miri-o/Documents/filtered_data_sets/Celiac_for_V_family_analysis_3M_seqs_Celiac_n_3_trimming_2_1_labeled_FILTERED_DATA.csv', sep = ',')

if len(vectors) == len(data):
    print('Data length validation succeeded')
else:
    print('Data validation FAILED...')
    sys.exit(1)


valid_indexes = data.index[~data.V_FAMILY.isnull()] # Remove rows where v_family is null 
sub_indexes = [*np.random.choice(data.index[data.V_FAMILY=='IGHV1'], 500000), *np.random.choice(data.index[data.V_FAMILY=='IGHV3'], 500000), *np.random.choice(data.index[data.V_FAMILY=='IGHV4'], 500000)]
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
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Logistic regression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print('Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(logreg.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
     .format(logreg.score(X_test, y_test)))

#Decision Tree

clf = DecisionTreeClassifier().fit(X_train, y_train)
print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))

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

pred = knn.predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

#Linear Discriminant Analysis

lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
print('Accuracy of LDA classifier on training set: {:.2f}'
     .format(lda.score(X_train, y_train)))
print('Accuracy of LDA classifier on test set: {:.2f}'
     .format(lda.score(X_test, y_test)))

#Gaussian Naive Bayes

gnb = GaussianNB()
gnb.fit(X_train, y_train)
print('Accuracy of GNB classifier on training set: {:.2f}'
     .format(gnb.score(X_train, y_train)))
print('Accuracy of GNB classifier on test set: {:.2f}'
     .format(gnb.score(X_test, y_test)))

## Support Vector Machine
svm = SVC()
svm.fit(X_train, y_train)
print('Accuracy of SVM classifier on training set: {:.2f}'
     .format(svm.score(X_train, y_train)))
print('Accuracy of SVM classifier on test set: {:.2f}'
     .format(svm.score(X_test, y_test)))

pred = svm.predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))


#https://towardsdatascience.com/solving-a-simple-classification-problem-with-python-fruits-lovers-edition-d20ab6b071d2