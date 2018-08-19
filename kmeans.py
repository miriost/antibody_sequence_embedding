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
import os


def plot_embedding_with_properties(x,y, prop, title=None, filename = None):
    # normalize property
    #prop = [(c - min(prop))/(max(prop)-min(prop))*100 for c in prop]
    
    sc = plt.scatter(x, y, c=prop, s=2, cmap='plasma', alpha = .7, marker = 'D')
    plt.colorbar(sc)
    plt.xlabel('1st component')
    plt.ylabel('2nd component')
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
    
    
    Attributes
    -------
    
    n_clusters:
        the total number of clusters would be n^depth (unless I will have a different criteria)

    
    
    """
    def __init__(self, n=20, depth=2, debug_mode = True, method = 'kmeans'):        
        self.n = n  
        self.depth = depth
        self.n_clusters_ = 0
        self.debug_mode = debug_mode
        self.method = method
        
    def fit(self, X, y=None):
        """
            X - a dataframe where each column is the features. X may contain index column (X.index)
        """
        self.X = X
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        
        t0 = time.time()
        if self.method == 'kmeans':
            iteration = 1
            if self.debug_mode:
                print('Starting {} clustering, iteration #{}, data size: {}'.format(self.method, iteration, len(X)))
            km = KMeans(n_clusters=self.n).fit(X)
            clusters = km.labels_
            clust_index = self.n
            while iteration < self.depth:
                clusters = {}
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
                            clusters[clust_index] = original_indexes[y_hat_small.labels_==j]
                            if self.debug_mode:
                                print('Cluster number {}:, {} points'.format(clust_index, len(clusters[clust_index])))
                                print('~~~ finished, time: {:.4} sec'.format(time.time()-t01))  
        #                log_file.write('~~~ finished, time: {:.4} sec'.format(time.time()-t0))
        #                    log_file.write('Cluster number {}:, {} points'.format(clust_index, len(clusters[clust_index])))
                            clust_index +=1   
                self.depth = self.depth-1
            
            self.n_clusters_ = clust_index 
        self.clusters_ = clusters
        self.clustering_time_ = time.time()-t0

        if self.debug_mode:
            print('TOTAL clustering time = {} seconds'.format(self.clustering_time_))
        #np.save('HCV_labels.npy', clusters)
        self.convert_clusters_to_labels()
        return self
    
    def convert_clusters_to_labels(self):
        """
        self.labels_ is dictionary, where each key is a cluster, and the values are the indexes contained in the cluster
        This function will contain
        """
        inv_clusters = {}
        for k in self.clusters_: #go over every cluster and inverse the values
            indexes = self.clusters_[k]
            for index in indexes:
                inv_clusters[index] = k
        self.labels_ = inv_clusters
        return(self)
        
    def create_feature_table(self, classes, TH = 100):
        """
        input - Classes, the true labels of the data (Data frame with indexes)
        TH - Threshold for filtering (Features where the precentage of observations from one subject > TH will be filtered)
        Build a feature table where each column is a cluster and each raw is a subject, 
        count the number of sequnces of each subject in each cluster
        """
            
        features_table = pd.DataFrame(0, index=pd.unique(classes), columns=self.clusters_.keys())
        #print(self.clusters_.keys())
        for index, row in enumerate(classes):
            #print(str(index) +' | '+ str(row)+' | '+ str(self.labels_[index]))
            features_table.loc[row, self.labels_[index]] +=1
        
        # Normlize by raw
        normlized_features_table = features_table.div(features_table.sum(axis=1), axis=0)
            
        ### Additional filtering - filter clusters with low subject diversity, i.e. X% of cell originated from one subject
        print('~~~ Filtering features with low diversity, TH = ' , str(TH))
        all_features_sum = normlized_features_table.sum(axis=0)
        all_features_max = normlized_features_table.max(axis=0)
        max_feature_precentage = all_features_max*100/all_features_sum
        max_feature_precentage = pd.DataFrame(max_feature_precentage, columns=['Precentage'])
        to_drop = pd.DataFrame([(center, value) for (center, value) in zip(max_feature_precentage.index, max_feature_precentage.Precentage) if value>TH], columns= ['Cluster', 'Precentage_of_top_feature'])
        
        # droping the above centers
        filtered_feature_table = normlized_features_table.drop(labels=to_drop.Cluster, axis = 1)
        print('Filtering {} columns where maximal feature precentage exceeds allowed TH, Centers: {}, Precentages: {}'.
              format(len(to_drop),list(to_drop.Cluster), list(to_drop.Precentage_of_top_feature)))
        self.feature_table = filtered_feature_table
        
        return self
    
    def save_feature_table(self, filename, path = None):
        name = os.path.split(filename)[-1].split(os.path.extsep)[0]
        final_dest = os.path.join(path, name+'_feature_table.csv')
        self.feature_table.to_csv(final_dest)
        print('Feature table saved to: ' + os.path.abspath(final_dest))
    
    def save_clusters_to_file(self, filename):      
        # Add cluster labels to existing file
        if os.path.isfile(filename):
            datafile = pd.read_csv(filename)
            datafile['cluster'] = [self.labels_[i] for i in datafile.index]
            datafile.to_csv(filename)
            print('Added "cluster" column to ' + filename)
        else:
            print('Error, miri please finis your code!')
#            indexes = sorted(self.labels_.keys())
#            values = [self.labels_[i] for i in indexes]
#            data_for_file = pd.DataFrame(values, index=indexes)
#            data_for_file.to_csv('/media/miri-o/Documents/Immune2vec/clusters/test1.csv')
#            print('Created new clusters file',filename)
        
    def save_labels_to_file(self, filename):
        pass
    
    def visualize(self):
        colors = [self.labels_[c] for c in self.X.index]
        plot_embedding_with_properties(self.X.iloc[:,0], self.X.iloc[:,1], colors, title = ('clustering'))
            
    