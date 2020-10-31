#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 16:52:40 2018

@author: miri-o
"""

import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import RFE
from scipy.stats import randint
from classifying.RepertoireClassifier import *


def rfc_features_selector(estimator, X_train, y_train, n_splits=10, repeated=True):

    data_x = X_train
    data_y = y_train
    feature_names = data_x.columns
    kf = KFold(n_splits=n_splits, shuffle=repeated)
    N = round(len(data_x.columns) * 0.3)
    all_importances = pd.DataFrame()
    for train, _ in kf.split(data_x, y_train):
        rfc = RandomForestClassifier(n_estimators=100)
        rfc.fit(data_x.iloc[ train, :], data_y[train])
        # sorting
        importances_index_desc = np.argsort(rfc.feature_importances_)[::-1]
        feature_labels = [feature_names[i] for i in importances_index_desc ]
        importances1 = pd.DataFrame({'FEATURE': data_x.columns, 'IMPORTANCE': np.round(rfc.feature_importances_, 3)})
        importances = importances1.sort_values('IMPORTANCE', ascending=False).set_index('FEATURE')
        # appending
        all_importances = all_importances.append(importances)

    all_importances_mean = all_importances.groupby('FEATURE').mean()
    all_importances_sort = all_importances_mean.sort_values('IMPORTANCE', ascending=False)
    # saving
    average_importance_features = list(all_importances_sort.index)
    # ~~~~~~~~~~~~~~~~ FS_RFE ~~~~~~~~~~~~~~~~~~~~~~~
    rf_features = average_importance_features[ 0:N ]
    X = X_train.loc[ :, rf_features ]
    rfe = RFE(estimator, n_features_to_select=round(N * 0.3), step=0.01)
    rfe = rfe.fit(X, y_train)

    # saving selected features
    index_feature = [ i for i, x in enumerate(rfe.support_) if x ]
    return X.columns.values[index_feature]


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


class Classifier:

    def __init__(self, name, classes, model=None, parameters=None):

        self.classes = classes
            
        # turn string labels to numeric labels
        self.class_dict = dict(zip(self.classes, range(len(self.classes))))

        self.classifier = None
        self.clf = None

        if model is not None:
            self.classifier = RepertoireClassifier(name, trained_model=model)
            return

        feature_selector = None

        if name.lower() in ['logistic_regression', 'lr']:
            if len(self.classes) == 2 and parameters is None: #binomial logistic regression case
                parameters = [
                    {'C': np.logspace(-3, 3, 7), 'class_weight': [None], 'dual': [False], 'fit_intercept': [True],
                     'max_iter': [100], 'multi_class': ['ovr'], 'penalty': ['l1', 'l2'], 'solver': ['liblinear'],
                     'warm_start': [False]}
                ]

            elif len(self.classes) > 2 and parameters is None: #Multinomial logistic regression
                parameters = [
                    {'C': np.logspace(-4, 4, 20), 'class_weight': ['balanced'], 'dual': [False],
                     'fit_intercept': [True], 'max_iter': [100], 'multi_class': ['multinomial'],
                     'penalty': ['l1', 'l2'], 'solver': ['newton-cg'], 'warm_start': [True]}
                ]
            feature_selector = rfc_features_selector
            estimator = LogisticRegression()

        elif name.lower() in [ 'decision_tree', 'dt' ]:
            if not parameters:
                parameters = [{'criterion': ['gini', 'entropy' ], 'max_depth': list(range(3, 15))}]
            estimator = DecisionTreeClassifier()

        elif name.lower() in ['knn', 'k-nn']:
            if not parameters:
                parameters = [{'n_neighbors': list(range(3, 11)), 'weights': ['uniform', 'distance'],
                               'algorithm': ['brute']}]
            estimator = KNeighborsClassifier()

        elif name.lower() in ['linear_svm', 'lsvm']:
            if not parameters:
                parameters = {'C': [0.1, 1, 10, 100, 1000], 'kernel': ['linear']}
            estimator = SVC()

        elif name.lower() in ['rbf_svm']:
            if not parameters:
                parameters = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']}
            estimator = SVC()

        elif name.lower() in ['gaussian']:
            if not parameters:
                parameters = {'kernel': [1.0 * RBF(1.0)]}
            estimator = GaussianProcessClassifier()
            
        elif name.lower() in ['rf', 'random_forest']:
            if not parameters:
                parameters = [{'max_depth': list(range(3,15))}]
            estimator = RandomForestClassifier()

        elif name.lower() in ['mlp', 'neural_net']:
            if not parameters:
                parameters = [{'max_iter': [1000]}]
            estimator = MLPClassifier()

        elif name.lower() in ['ada', 'adaboost', 'ada_boost']:
            if not parameters:
                parameters = [{}]
            estimator = AdaBoostClassifier()

        elif name.lower() in ['NB', 'naive_bayes', 'Naive_Bayes', 'Naive_bayes' ]:
            if not parameters:
                parameters = [{'var_smoothing': [n*1e-9 for n in range(1, 5)]}]
            estimator = GaussianNB()

        elif name.lower() in ['qda']:
            if not parameters:
                parameters = [{}]
            estimator = QuadraticDiscriminantAnalysis()
                
        else:
            raise Exception("Classifier model un-recognized, current supported models: logistic_regression, decision_tree, kNN, linear_svm, RBF_SVM, Gaussian, Random_Forest, MLP, ADA, naive_bayes, QDA")

        self.classifier = RepertoireClassifier(name,
                                               estimator=estimator,
                                               feature_selector=feature_selector,
                                               parameters=parameters)


    def run_once(self, X_train, X_test, y_train, y_test):

        self.classifier.select_features(X_train, y_train)
        self.classifier.fit(X_train, y_train)
        train_predictions = self.classifier.predict(X_train)
        test_predictions = self.classifier.predict(X_test)

        train_report = classification_report(y_train, train_predictions, zero_division=0, output_dict=True)
        test_report = classification_report(y_test, test_predictions, zero_division=0, output_dict=True)
        train_confusion_matrix = confusion_matrix(y_train, train_predictions, labels=self.classes)
        test_confusion_matrix = confusion_matrix(y_test, test_predictions, labels=self.classes)

        idx = 0
        accuracy = 0
        output = pd.DataFrame(columns=['confusion_matrix'])

        for key, item in train_report.items():
            if key == 'accuracy':
                accuracy = item
                continue
            output.loc[idx, 'precision'] = item['precision']
            output.loc[idx, 'recall'] = item['recall']
            output.loc[idx, 'f1_score'] = item['f1-score']
            output.loc[idx, 'key'] = key
            if key in self.classes:
                class_idx = np.where(np.array(self.classes) == key)[0][0]
                output.at[idx, 'confusion_matrix'] = train_confusion_matrix[class_idx].tolist()
            else:
                output.at[idx, 'confusion_matrix'] = train_confusion_matrix.tolist()
            idx = idx + 1

        output['report'] = 'train'
        output['accuracy'] = accuracy
        output['confusion_matrix'] = [confusion_matrix(y_train, train_predictions).tolist()] * len(output)
        start = idx

        for key, item in test_report.items():
            if key == 'accuracy':
                accuracy = item
                continue
            output.loc[idx, 'precision'] = item['precision']
            output.loc[idx, 'recall'] = item['recall']
            output.loc[idx, 'f1_score'] = item['f1-score']
            output.loc[idx, 'key'] = key
            if key in self.classes:
                class_idx = np.where(np.array(self.classes) == key)[0][0]
                output.at[idx, 'confusion_matrix'] = test_confusion_matrix[class_idx].tolist()
            else:
                output.at[idx, 'confusion_matrix'] = test_confusion_matrix.tolist()
            idx = idx + 1

        output.loc[list(range(start, idx)), 'report'] = 'test'
        output.loc[list(range(start, idx)), 'accuracy'] = accuracy

        output['n_features'] = len(self.classifier.features)
        output['model'] = self.classifier.name
        output['parameters'] = format(self.classifier.parameters)
        output['labels'] = format(self.classes.tolist())

        # sort columns
        output = output.reindex(sorted(output.columns), axis=1)

        return output, self.classifier.trained_model

