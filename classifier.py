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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


def add(a,b):
    return(a+b)

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
    
    """ Perform a classification based on a feature matrix
 
    Parameters
    ----------       

    depth : integer, defalut 2
        the number of k-means iterations. CURRENTLY SUPPORTING ONLY DEPTH OF 2.
        
    visualize : Boolean, default False
        whether or not to visualize the clustering (relevant to 2D clustering)
    
    
    Attributes
    -------
  
    
    """
    
    def __init__(self, feature_table, labels, model, classes = None, C = 1.0):
        self.feature_table = feature_table
        self.labels = labels
        self.modelname = model 
        self.coef = np.zeros(feature_table.shape[1])
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
        
        model_names = ["Nearest Neighbors (kNN)", "Linear SVM (LSVM)", "RBF SVM (RBF_SVM)", "Gaussian Process (Gaussian)",
         "Decision Tree (DT)", "Random Forest (RF)", "Neural Net (MLP)", "AdaBoost ('Ada')",
         "Naive Bayes (NB)", "QDA"]

        
        if self.modelname == 'logistic_regression' or self.modelname == 'LR':
            if len(self.classes) == 2: #binomial logistic regression case
                self.model = LogisticRegression(C=C, class_weight=None, dual=False, fit_intercept=True,
                                                intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
                                                penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
                                                verbose=0, warm_start=False)
            else: #Multinomial logistic regression
                self.model = LogisticRegression(C=C, class_weight='balanced', dual=False, fit_intercept=True, 
                                                intercept_scaling=1, max_iter=100, multi_class='multinomial', n_jobs=1,
                                                penalty='l2', random_state=None, solver='newton-cg', tol=0.0001,
                                                verbose=0, warm_start=True)
        elif self.modelname == 'regulized_logistic_regression' or self.modelname == 'RLR':
            if len(self.classes) == 2: #binomial logistic regression case
                self.model = LogisticRegression(C=C, class_weight=None, dual=False, fit_intercept=True,
                                                intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
                                                penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
                                                verbose=0, warm_start=False)
            else: #Multinomial logistic regression
                self.model = LogisticRegression(C=C, class_weight='balanced', dual=False, fit_intercept=True, 
                                                intercept_scaling=1, max_iter=100, multi_class='multinomial', n_jobs=1,
                                                penalty='l2', random_state=None, solver='newton-cg', tol=0.0001,
                                                verbose=0, warm_start=True)
        elif self.modelname in ['decision_tree','DT']:
            self.model = tree.DecisionTreeClassifier(max_depth=5)
            
        elif self.modelname in ['kNN','k-NN','knn']:
            self.model = KNeighborsClassifier(n_neighbors=3)
            
        elif self.modelname in ['linear_svm', 'LSVM']:
            self.model = SVC(kernel="linear", C=0.025)
        
        elif self.modelname in ['rbf_svm', 'RBF_SVM']:
            self.model = SVC(gamma=2, C=1)
        
        elif self.modelname in ['Gaussian', 'gaussian']:
            self.model = GaussianProcessClassifier(1.0 * RBF(1.0))
            
        elif self.modelname in ['RF', 'Random_forest', 'Random_Forest', 'random_forest']:
            self.model = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
            
        elif self.modelname in ['MLP', 'Neural_net']:
            self.model = MLPClassifier(alpha=1, max_iter=1000)
        
        elif self.modelname in ['ADA', 'Ada', 'Adaboost', 'Ada_boost']:
            self.model = AdaBoostClassifier()
            
        elif self.modelname in ['NB', 'naive_bayes','Naive_Bayes','Naive_bayes']:
            self.model = GaussianNB()
        
        elif self.modelname in ['QDA','qda']:
            self.model = QuadraticDiscriminantAnalysis()
                
        else:
            raise Exception("Classifier model un-recognized, current supported models: logistic_regression, decision_tree, kNN, linear_svm, RBF_SVM, Gaussian, Random_Forest, MLP, ADA, MLP, naive_bayes, QDA")
      
    
    def run(self, n = 1000, test_size = .2):
        predictions_all =[]
        actual_all = []
        
        for i in range(n):
            X_train, X_test, y_train, y_test = train_test_split(self.feature_table, self.labels_num, test_size=test_size)
            
            self.model.fit(X_train,y_train)
            if self.modelname in ["LR", "RLR", 'logistic_regression', 'regulized_logistic_regression']:
                self.coef = list(map(add, self.coef, self.model.coef_))
            predictions_all.extend(list(self.model.predict(X_test)))
            actual_all.extend(y_test)
            #coefs = coefs + self.model.coef_
         
    # Compute confusion matrix
        cnf_matrix = confusion_matrix(actual_all, predictions_all)
#        print('actual:')
#        print(actual_all)
#        print('prediction:')
#        print()
        np.set_printoptions(precision=2)
        
        plot_confusion_matrix(cnf_matrix, classes=self.classes, normalize=True,
                          title='Normalized confusion matrix'+ ' score: ' + str(accuracy_score(actual_all, predictions_all)))
        plt.show()
        self.score = accuracy_score(actual_all, predictions_all)
        print('classifier' + self.modelname +'score: ' + "%.3f" % self.score)
        
        if self.model == 'decision_tree' or self.model == 'DT':
            dot_data = tree.export_graphviz(self.model, out_file='../../results/tmp.dot', 
                                 feature_names=self.feature_table.columns,  
                                 class_names=self.classes,  
                                 filled=True, rounded=True,  
                                 special_characters=True)  
    #        graph = graphviz.Source(dot_data)  
#        graph.show()
        return(self)

