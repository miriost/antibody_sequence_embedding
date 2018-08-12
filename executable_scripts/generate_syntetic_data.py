# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 19:59:58 2018

@author: mirio

Generate syntetic data set for flow testing 
"""

## Healthy group

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

group_size = 50
subject_size = 500
board_range = (50, 50)
DF = pd.DataFrame(columns = ['x','y','subject', 'condition'])
DF_for_clustering = pd.DataFrame(columns = ['x','y','labels'])
#plt.figure()
for j in range(group_size):
    x = np.random.normal(np.random.choice(board_range[0]), scale=1.0, size=500)
    y = np.random.normal(np.random.choice(board_range[1]), scale=1.0, size=500)
    subject = ['H'+str(j) for i in range(len(x))]
    condition = ['H' for i in range(len(x))]
    DF2 = pd.DataFrame({'x':x, 'y':y, 'subject':subject, 'condition':condition})
    DF_c = pd.DataFrame({'x':x, 'y':y, 'labels':subject})
    DF_for_clustering = DF_for_clustering.append(DF_c)
    DF = DF.append(DF2, ignore_index=True)
    #plt.scatter(x, y, s=5, marker = '^')
    
for j in range(group_size):
    x = np.random.normal(np.random.choice(board_range[0]), scale=1.0, size=500)
    y = np.random.normal(np.random.choice(board_range[1]), scale=1.0, size=500)
    subject = ['S'+str(j) for i in range(len(x))]
    condition = ['S' for i in range(len(x))]
    DF_c = pd.DataFrame({'x':x, 'y':y, 'labels':subject})
    DF_for_clustering = DF_for_clustering.append(DF_c)
    DF2 = pd.DataFrame({'x':x, 'y':y, 'subject':subject, 'condition':condition})
    DF = DF.append(DF2, ignore_index = True)
    #plt.scatter(x, y, s=5, marker = 'o')

print('length:' + str(len(DF)))
print(DF.describe)
path = os.path.join(os.pardir, os.path.pardir, 'results')
final_dest = os.path.join(path, str(group_size*2)+'_syntetic_data.csv')
DF.to_csv(final_dest)
print('syntetic_data saved to: ' + os.path.abspath(final_dest))

path = os.path.join(os.pardir, os.path.pardir, 'results')
final_dest = os.path.join(path, str(group_size*2)+'_syntetic_vectors.csv')
DF_for_clustering.to_csv(final_dest, index = None)
print('syntetic vectors saved to: ' + os.path.abspath(final_dest))