#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 14:25:40 2018

@author: miri-o
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#matr = np.random.random((100, 2))
#data = pd.DataFrame(matr)
#data.to_csv('/media/miri-o/Documents/Immune2vec/vectors/random_matrix_100x2.csv')
dist1x = np.random.normal(1, scale=1.0, size=500)
dist1y = np.random.normal(1, scale=1.0, size=500)
plt.scatter(dist1x, dist1y)

dist2x = np.random.normal(3, scale=0.7, size=500)
dist2y = np.random.normal(.6, scale=1.3, size=500)
plt.scatter(dist2x, dist2y)

dist3x = np.random.normal(6, scale=0.5, size=500)
dist3y = np.random.normal(5, scale=1.4, size=500)
plt.scatter(dist3x, dist3y)

dist4x = np.random.normal(6, scale=1.7, size=500)
dist4y = np.random.normal(10, scale=.8, size=500)
plt.scatter(dist4x, dist4y)

x = np.concatenate((dist1x, dist2x, dist3x, dist4x))
y = np.concatenate((dist1y, dist2y, dist3y, dist4y))
labels = np.concatenate((np.ones(500), np.ones(500)*2, np.ones(500)*3, np.ones(500)*4))

data = {'x':x, 'y':y, 'labels':labels}
DF = pd.DataFrame(data, columns=['x','y', 'labels'])
DF.to_csv('/media/miri-o/Documents/Immune2vec/vectors/4_gaussians_test.csv', index=False)
#print(DF)



