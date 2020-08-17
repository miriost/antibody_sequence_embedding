# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 10:56:02 2020

@author: mirio
"""
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

dimensions= [0, 2, 5, 10, 15, 20, 30, 50, 75, 90, 100]
knn_f1_score= [0.6004093567251462, 0.6337894736842106, 0.6553216374269006, 0.6627368421052632, 0.670374269005848, 0.6781403508771929, 0.6845730994152047, 0.6803625730994152, 0.6782807017543859, 0.6736608187134503]
decision_tree_f1_score= [0.6621520467836257, 0.6835789473684211, 0.6864912280701755, 0.6870877192982456, 0.693204678362573, 0.6895321637426901, 0.6904210526315789, 0.6910643274853802, 0.6886900584795321, 0.688233918128655]


knn_f1_score.insert(0,0)
decision_tree_f1_score.insert(0,0)

#sns.set(color_codes=True)
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plt.plot(dimensions, decision_tree_f1_score, lw=2, marker='d',markersize=7,alpha = 0.8)
plt.ylim(0,1)
plt.xlim(-2,100)
plt.plot(dimensions, knn_f1_score, color = 'r', linewidth=2, marker='o',alpha = 0.8, markersize=7)
plt.legend(['Decision tree', 'k-NN'])
plt.title('f1-score of V family classifiers', fontsize = 16)
plt.xlabel('Number of dimensions', fontsize = 14)
plt.ylabel('f1-score', fontsize = 14)
plt.show()
