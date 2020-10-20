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
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, recall_score
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
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

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
    
    def __init__(self, feature_table, labels, model_name, classes=None, C=1.0, grid_search=True, model=None):
        self.feature_table = feature_table
        self.labels = labels
        self.model_name = model_name
        self.coef = np.zeros(feature_table.shape[1])
        if len(feature_table)!=len(labels):
            raise Exception("Feature table and labels length mismatch!")
        if classes:
            self.classes = classes
        else:
            self.classes = np.unique(labels)
            
        # turn string labels to numeric labels
        self.class_dict = dict(zip(self.classes, range(len(self.classes))))
        self.labels_num = pd.Series(self.labels).map(self.class_dict, na_action='ignore')
#        print(self.labels_num)
#        print(self.labels)
        
        model_names = ["Nearest Neighbors (kNN)", "Linear SVM (LSVM)", "RBF SVM (RBF_SVM)", "Gaussian Process (Gaussian)",
         "Decision Tree (DT)", "Random Forest (RF)", "Neural Net (MLP)", "AdaBoost ('Ada')",
         "Naive Bayes (NB)", "QDA"]

        self.models = None
        self.model = None
        self.clf = None
        self.trained_model = None

        if model is not None:
            self.trained_model = model
            return

        if self.model_name == 'logistic_regression' or self.model_name == 'LR':
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
        elif self.model_name == 'regulized_logistic_regression' or self.model_name == 'RLR':
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

        elif self.model_name in [ 'decision_tree', 'DT' ]:
            if not grid_search:
                self.model = DecisionTreeClassifier(max_depth=3)
            else:
                self.models = [(DecisionTreeClassifier(max_depth=depth),
                                {'max_depth': depth}) for depth in range(3, 15)]

        elif self.model_name in [ 'kNN', 'k-NN', 'knn' ]:
            if not grid_search:
                tuned_parameters = [{'n_neighbors': [3]}]
            else:
                tuned_parameters = [{'n_neighbors': list(range(3, 11)),
                                     'weights': ['uniform', 'distance'],
                                     'algorithm': ['brute']}]
            self.clf = GridSearchCV(KNeighborsClassifier(), tuned_parameters, scoring='f1_macro')

        elif self.model_name in [ 'linear_svm', 'LSVM' ]:
            if not grid_search:
                tuned_parameters = {'C': [100], 'kernel': ['linear']}
            else:
                tuned_parameters = {'C': [0.1, 1, 10, 100, 1000], 'kernel': ['linear']}
            self.clf = GridSearchCV(SVC(), tuned_parameters, scoring='f1_macro')

        elif self.model_name in [ 'rbf_svm', 'RBF_SVM' ]:
            if not grid_search:
                tuned_parameters = {'C': [100], 'gamma': [1], 'kernel': ['rbf']}
            else:
                tuned_parameters = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']}
            self.clf = GridSearchCV(SVC(), tuned_parameters, scoring='f1_macro')

        elif self.model_name in [ 'Gaussian', 'gaussian' ]:
            self.model = GaussianProcessClassifier(1.0 * RBF(1.0))
            
        elif self.model_name in [ 'RF', 'Random_forest', 'Random_Forest', 'random_forest' ]:
            if not grid_search:
                self.model = RandomForestClassifier()
            else:
                tuned_parameters = [{'max_depth': randint(3, 14), 'max_features': randint(1, 5)}]
                self.clf = RandomizedSearchCV(RandomForestClassifier(), tuned_parameters, scoring='f1_macro')

        elif self.model_name in [ 'MLP', 'Neural_net' ]:
            if not grid_search:
                self.model = MLPClassifier(max_iter=1000)
            else:
                tuned_parameters = {
                    'activation': ['tanh', 'relu'],
                    'solver': ['sgd', 'adam'],
                    'alpha': [0.0001, 0.05],
                    'learning_rate': ['constant', 'adaptive'],
                }
                self.clf = GridSearchCV(MLPClassifier(max_iter=1000), tuned_parameters, scoring='f1_macro')

        elif self.model_name in [ 'ADA', 'Ada', 'Adaboost', 'Ada_boost' ]:
            self.model = AdaBoostClassifier()

        elif self.model_name in [ 'NB', 'naive_bayes', 'Naive_Bayes', 'Naive_bayes' ]:
            if not grid_search:
                self.model = GaussianNB()
            else:
                tuned_parameters = [{'var_smoothing': [n*1e-9 for n in range(1, 5)]}]
                self.clf = GridSearchCV(GaussianNB(), tuned_parameters, scoring='f1_macro')

        elif self.model_name in [ 'QDA', 'qda' ]:
            self.model = QuadraticDiscriminantAnalysis()
                
        else:
            raise Exception("Classifier model un-recognized, current supported models: logistic_regression, decision_tree, kNN, linear_svm, RBF_SVM, Gaussian, Random_Forest, MLP, ADA, naive_bayes, QDA")

    def run(self, n = 1000, test_size = .2, validate=False, X_validate=None, y_validate=None):
        predictions_all =[]
        actual_all = []
        if validate and (X_validate is None or y_validate is None):
            raise Exception("validation data is missing")
        else:
            print('Beginning cross correlation including validation set')
            validate_prediction = []
            validate_actual = []
            y_validate_num = pd.Series(y_validate).map(self.class_dict, na_action='ignore')
        
        
        for i in range(n):
            X_train, X_test, y_train, y_test = train_test_split(self.feature_table, self.labels_num, test_size=test_size)

            if self.model is not None:
                self.model.fit(X_train, y_train)
                validata_pred = self.model.predict(X_validate)
                test_preds = self.model.predict(X_test)
            elif self.clf is not None:
                validata_pred = self.clf.predict(X_validate)
                test_preds = self.clf.predict(X_test)
            else:
                return(self)

            if self.model_name in [ "LR", "RLR", 'logistic_regression', 'regulized_logistic_regression' ]:
                self.coef = list(map(add, self.coef, self.model.coef_))
            predictions_all.extend(list(test_preds))
            actual_all.extend(y_test)
            if validate:
                validate_prediction.extend(list(validata_pred))
                validate_actual.extend(y_validate_num)
            #coefs = coefs + self.model.coef_
         
        # Compute confusion matrix
        cnf_matrix = confusion_matrix(actual_all, predictions_all)
        np.set_printoptions(precision=2)        
        plot_confusion_matrix(cnf_matrix, classes=self.classes, normalize=True,
                          title='Normalized confusion matrix'+ ' score: ' + str(accuracy_score(actual_all, predictions_all)))
        plt.show()
        print(actual_all[:40])
        print(predictions_all[:40])
        self.score = [accuracy_score(actual_all, predictions_all), f1_score(actual_all, predictions_all), precision_score(actual_all, predictions_all), recall_score(actual_all, predictions_all)]
        print('Classifier: {} scores (Accuracy, f1, precision, recall) - {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(self.model_name, self.score[0 ], self.score[1 ], self.score[2 ], self.score[3 ]))
        if validate:
            print(validate_actual[:40])
            print(validate_prediction[:40])
            self.validation_score = [accuracy_score(validate_actual, validate_prediction), f1_score(validate_actual, validate_prediction), precision_score(validate_actual, validate_prediction), recall_score(validate_actual, validate_prediction)]
            print('Classifier: {} scores (Accuracy, f1, precision, recall) - {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(self.model_name, self.validation_score[0 ], self.validation_score[1 ], self.validation_score[2 ], self.validation_score[3 ]))
            
        #print('classifier ' + self.modelname +' score: ' + "%.3f" % self.score)
        
        if self.model is not None and self.model_name == 'decision_tree' or self.model_name == 'DT':
            dot_data = tree.export_graphviz(self.model, out_file='../../results/tmp.dot',
                                 feature_names=self.feature_table.columns,  
                                 class_names=self.classes,  
                                 filled=True, rounded=True,  
                                 special_characters=True)  
    #        graph = graphviz.Source(dot_data)  
#        graph.show()
        return(self)

    def run_once(self, X_train, X_test, y_train, y_test):

        parameters = {}
        model = None

        if self.trained_model is not None:
            model = self.trained_model
            train_predictions = model.predict(X_train)
            test_predictions = model.predict(X_test)

        elif self.model is not None:
            # Train our classifier
            model = self.model
            model.fit(X_train, y_train)
            # Make predictions
            train_predictions = model.predict(X_train)
            test_predictions = model.predict(X_test)
        
        elif self.models is not None:
            best_accuracy_score = 0
            train_predictions = []
            test_predictions = []
            for model_it in self.models:
                model_it[0].fit(X_train, y_train)
                # Make predictions
                tmp_train_predictions = model_it[0].predict(X_train)
                tmp_test_predictions = model_it[0].predict(X_test)
                if accuracy_score(y_test, tmp_test_predictions) >= best_accuracy_score:
                    model = model_it[0]
                    best_accuracy_score = accuracy_score(y_test, tmp_test_predictions)
                    train_predictions = tmp_train_predictions
                    test_predictions = tmp_test_predictions
                    parameters = model_it[1]
        
        elif self.clf is not None:
            self.clf.fit(X_train, y_train)
            model = self.clf.best_estimator_
            parameters = self.clf.best_params_
            train_predictions = model.predict(X_train)
            test_predictions = model.predict(X_test)
        
        else:
            # return empty data frame
            return pd.DataFrame()

        train_report = classification_report(y_train, train_predictions, zero_division=0, output_dict=True)
        test_report = classification_report(y_test, test_predictions, zero_division=0, output_dict=True)

        output = pd.DataFrame()
        idx = 0
        accuracy = 0
        for key, item in train_report.items():
            if key == 'accuracy':
                accuracy = item
                continue
            output.loc[idx, 'precision'] = item['precision']
            output.loc[idx, 'recall'] = item['recall']
            output.loc[idx, 'f1_score'] = item['f1-score']
            output.loc[idx, 'key'] = key
            idx = idx + 1

        output['report'] = 'train'
        output['accuracy'] = accuracy
        start = idx

        for key, item in test_report.items():
            if key == 'accuracy':
                accuracy = item
                continue
            output.loc[idx, 'precision'] = item['precision']
            output.loc[idx, 'recall'] = item['recall']
            output.loc[idx, 'f1_score'] = item['f1-score']
            output.loc[idx, 'key'] = key
            idx = idx + 1

        output.loc[list(range(start, idx)), 'report'] = 'test'
        output.loc[list(range(start, idx)), 'accuracy'] = accuracy

        output['n_features'] = X_train.shape[1]
        output['model'] = self.model_name
        output['parameters'] = format(parameters)

        # sort columns
        output = output.reindex(sorted(output.columns), axis=1)

        return output, model
