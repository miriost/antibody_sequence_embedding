#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 12:43:25 2018

@author: miri-o
"""
#import numpy as np
import time
#import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def plot_embedding_with_properties(x,y, prop, title=None, filename = None):
    # normalize property
    #prop = [(c - min(prop))/(max(prop)-min(prop))*100 for c in prop]
    
    sc = plt.scatter(x, y, c=prop, s=2, cmap='plasma', alpha = .7, marker = 'D')
    plt.colorbar(sc)
    plt.xlabel('tSNE component 1')
    plt.ylabel('tSNE component 2')
    #plt.axis([xmin, xmax, ymin, ymax])
    if title is not None:
        plt.title(title) 
    if filename is not None:
        plt.savefig(filename+'.pdf', bbox_inches='tight')

    plt.show()

def kMeans_clustering(X, n, plot = False, path = None):
    print('~~~starting k-means clustering, N = ' + str(len(X)))
    t0 = time.time()
    kmeans = KMeans(n_clusters=n).fit(X)
    print('finished, time: {:.4} sec'.format(time.time()-t0)) 
    if plot:
        plot_embedding_with_properties(X[:,0], X[:,1], kmeans.labels_, title = 'k-means clustering', filename=path)
    return(kmeans)
