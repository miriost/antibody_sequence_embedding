#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 16:52:40 2018

@author: miri-o
"""

## confusion matrix plotting function for all the below models
import itertools
#from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import tree

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

#    plt.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
#    plt.title(title)
#    plt.colorbar()
#    tick_marks = np.arange(len(classes))
#    plt.xticks(tick_marks, classes, rotation=45)
#    plt.yticks(tick_marks, classes)
#
#    fmt = '.2f' if normalize else 'd'
#    thresh = cm.max() / 2.
#    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#        plt.text(j, i, format(cm[i, j], fmt),
#                 horizontalalignment="center",
#                 color="white" if cm[i, j] > thresh else "black")
#
#    plt.tight_layout()
#    plt.ylabel('True label')
#    plt.xlabel('Predicted label')
    

class classifier():
    
    """ Perform a classification based ona feature matrix
 
    Parameters
    ----------       

    depth : integer, defalut 2
        the number of k-means iterations. CURRENTLY SUPPORTING ONLY DEPTH OF 2.
        
    visualize : Boolean, default False
        whether or not to visualize the clustering (relevant to 2D clustering)
    
    
    Attributes
    -------
  
    
    """
    
    def __init__(self, feature_table, labels, model, classes = None):
        self.feature_table = feature_table
        self.labels = labels
        if (len(feature_table)!=len(labels)):
            raise Exception("Feature table and labels length mismatch!")
        if classes:
            self.classes = classes
        else:
            self.classes = np.unique(labels)
            
        # turn string labels to numeric labels
        d = dict(zip(self.classes, range(len(self.classes))))
        self.labels_num = pd.Series(self.labels).map(d, na_action='ignore')
#        print(self.labels_num)
#        print(self.labels)
        
        if model == 'logistic_regression':
            if len(self.classes) == 2: #binomial logistic regression case
                self.model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                                                intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
                                                penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
                                                verbose=0, warm_start=False)
            else: #Multinomial logistic regression
                self.model = LogisticRegression(C=1.0, class_weight='balanced', dual=False, fit_intercept=True, 
                                                intercept_scaling=1, max_iter=100, multi_class='multinomial', n_jobs=1,
                                                penalty='l2', random_state=None, solver='newton-cg', tol=0.0001,
                                                verbose=0, warm_start=True)
        elif model == 'decision_tree':
            self.model = tree.DecisionTreeClassifier(max_depth=3)
        else:
            raise Exception("Classifier model un-recognized, current supported models: logistic_regression, decision_tree")
      
    
    def run(self, n=1000, test_size=.2):
        predictions_all =[]
        actual_all = []
        
        for i in range(n):
            X_train, X_test, y_train, y_test = train_test_split(self.feature_table, self.labels_num, test_size=test_size)
            
            self.model.fit(X_train,y_train)
            predictions_all.extend(list(self.model.predict(X_test)))
            actual_all.extend(y_test)
            #coefs = coefs + self.model.coef_
         
    # Compute confusion matrix
        cnf_matrix = confusion_matrix(actual_all, predictions_all)
        np.set_printoptions(precision=2)
        
        plot_confusion_matrix(cnf_matrix, classes=self.classes, normalize=True,
                          title='Normalized confusion matrix'+ ' score: ' + str(accuracy_score(actual_all, predictions_all)))
        plt.show()
        self.score = accuracy_score(actual_all, predictions_all)
        print('score: ' + "%.3f" % self.score)
        
        if self.model == 'decision_tree':
            dot_data = tree.export_graphviz(self.model, out_file='../../results/tmp.dot', 
                                 feature_names=self.feature_table.columns,  
                                 class_names=self.classes,  
                                 filled=True, rounded=True,  
                                 special_characters=True)  
    #        graph = graphviz.Source(dot_data)  
#        graph.show()

