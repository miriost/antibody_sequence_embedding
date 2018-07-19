#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 12:43:25 2018

@author: miri-o
"""
#import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np


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

#def kMeans_clustering(X, n, plot = False, path = None):
#    print('~~~starting k-means clustering, N = ' + str(len(X)))
#    t0 = time.time()
#    kmeans = KMeans(n_clusters=n).fit(X)
#    print('finished, time: {:.4} sec'.format(time.time()-t0)) 
#    if plot:
#        plot_embedding_with_properties(X[:,0], X[:,1], kmeans.labels_, title = 'k-means clustering', filename=path)
#    return(kmeans)

class unsupervised_clustering():
    """ Perform k-means clustering of N-dimensional array
 
    Parameters
    ----------       
    n : integer, default 20
        the number of clusters per iteration
    
    depth : integer, defalut 2
        the number of k-means iterations. CURRENTLY SUPPORTING ONLY DEPTH OF 2.
        
    visualize : Boolean, default False
        whether or not to visualize the clustering (relevant to 2D clustering)
    
    filename: string, default None
        output file name to save the clustering labels
    
    
    Attributes
    -------
    
    n_clusters:
        the total number of clusters would be n^depth (unless I will have a different criteria)

    
    
    """
    def __init__(self, n=20, depth=2, visualize = False, filename = None, debug_mode = True, method = 'kmeans'):        
        self.n = n  
        self.depth = depth
        self.n_clusters_ = 0
        self.debug_mode = debug_mode
        self.filename = filename
        self.method = method
        self.visualize = visualize
        
    def fit(self, X, y=None):
        """
            X - a dataframe where each column is the features. X may contain index column (X.index)
        """
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        
        t0 = time.time()
        if self.method == 'kmeans':
            iteration = 1
            if self.debug_mode:
                print('Starting {} clustering, iteration #{}, data size: {}'.format(self.method, iteration, len(X)))
            km = KMeans(n_clusters=self.n).fit(X)

            labels = km.labels_
            clust_index = self.n
            while iteration < self.depth:
                labels = {}
                clust_index = 0
                if self.debug_mode:
                    print('finished, time: {:.4} sec'.format(time.time()-t0))
                    # log_file.write('finished, time: {:.4} sec'.format(time.time()-t0))
                    print('Starting SUB-CLUSTERING clustering with n='+str(self.n)+'\n')
                    # log_file.write('Starting SUB-CLUSTERING clustering with n='+str(n)+'\n')
                for i in range(km.n_clusters):
                    if self.debug_mode:
                        print('~~~Cluster number: {}, initial cluster size: {}'.format(i,np.sum(km.labels_==i)))
                    if np.sum(km.labels_==i)>10: #split only clusters larger than 10
                        original_indexes, = np.where(km.labels_==i)
                        X_small = X.loc[original_indexes] 
        #                log_file.write('~~~ Starting clustering of N='+str(len(X_small)))
                        t01 = time.time()
                        y_hat_small =  KMeans(n_clusters=self.n).fit(X_small)
                        for j in range(y_hat_small.n_clusters):
                            labels[clust_index] = original_indexes[y_hat_small.labels_==j]
                            if self.debug_mode:
                                print('Cluster number {}:, {} points'.format(clust_index, len(labels[clust_index])))
                                print('~~~ finished, time: {:.4} sec'.format(time.time()-t01))  
        #                log_file.write('~~~ finished, time: {:.4} sec'.format(time.time()-t0))
        #                    log_file.write('Cluster number {}:, {} points'.format(clust_index, len(labels[clust_index])))
                            clust_index +=1   
                self.depth = self.depth-1
            
            self.n_clusters_ = clust_index 
        self.labels_ = labels
        self.clustering_time_ = time.time()-t0

        if self.debug_mode:
            print('TOTAL clustering time = {} seconds'.format(self.clustering_time_))
        #np.save('HCV_labels.npy', labels)
        inv_labels = {}
        for k in labels.keys(): #go over every cluster and inverse the values
            indexes = labels[k]
            for index in indexes:
                inv_labels[index] = k
        self.clusters_ = inv_labels
        if self.visualize:
            colors = [self.clusters_[c] for c in range(len(X))]
            sc = plot_embedding_with_properties(X.iloc[:,0], X.iloc[:,1], colors, title = ('clustering'))
        if self.filename:        
        # insert the clusters to data file         
            datafile = pd.read_csv(self.filename, delimiter='\t')
            datafile['original index'] = X.index
            datafile['cluster'] = [inv_labels[i] for i in datafile['original index']]
        
        return self
