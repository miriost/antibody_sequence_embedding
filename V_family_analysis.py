#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 15:31:14 2018

@author: miri-o
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

file = pd.read_csv('/media/miri-o/Documents/filtered_data_sets/Celiac_for_V_family_analysis_3M_seqs_Celiac_n_3_trimming_2_1_FILTERED_DATA.csv',sep=',')
labels = pd.read_csv('/media/miri-o/Documents/filtered_data_sets/Celiac_for_V_Family_V_family_labels.csv', sep='\t')
valid_labels = ['NA' if ',' in y else y for y in labels.x ]
file['V_FAMILY'] = valid_labels
file.to_csv('/media/miri-o/Documents/filtered_data_sets/Celiac_for_V_family_analysis_3M_seqs_Celiac_n_3_trimming_2_1_labeled_FILTERED_DATA.csv', sep = ',')

# Now let's try straight-forawrd classification on our vectors based on the v family
vectors = pd.read_csv('/media/miri-o/Documents/vectors/Celiac_for_V_family_analysis_3M_seqs_Celiac_n_3_trimming_2_1_VECTORS_after_PCACeliac_10D.csv')

valid_indexes = [i for i,x in enumerate(valid_labels) if x!='NA']
X = vectors[valid_indexes]
y= valid_labels[valid_indexes]
# Create Training and Test Sets and Apply Scaling

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#https://towardsdatascience.com/solving-a-simple-classification-problem-with-python-fruits-lovers-edition-d20ab6b071d2