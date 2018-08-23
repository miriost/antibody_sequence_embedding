#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 11:35:40 2018

@author: miri-o

Dimensionlity reductin object, using 2 possible methods:
    1. PCA - n for number of dimesnions required
    2. tSNE - 2 or 3 dimensions (default 2)
    
Input - Matrix of word2vec vectors
    
"""
import sys
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import time


def standardPCA(data, ncomp):
        
    #Standardize the Data:
    s_data = StandardScaler().fit_transform(data)

    # Apply the PCA
    pca = PCA(n_components=ncomp)
    pca.fit(s_data)
    data_after_pca = pca.transform(s_data)
    print('Explained variance: {}, summing to {} of the data'.format(pca.explained_variance_ratio_, sum(pca.explained_variance_ratio_)))
    return(data_after_pca)


class dimensional_reduction():
    
    def __init__(self, input_file, output_file, n_dim, method = 'PCA'):
        self.n_dim = n_dim
        self.input_file = input_file
        self.output_file = output_file
        self.method = method
        
        if self.method == 'PCA':
            print('Starting PCA, reducing to n= {} dimensions'.format(self.n_dim))
            input_data = pd.read_csv(self.input_file)
            t0 = time.time()
            vec_embedded = standardPCA(input_data, self.n_dim)
            print('Embedding finished in {:.3} minutes'.format((time.time()-t0)/60))
            pd.DataFrame(vec_embedded).to_csv(self.output_file, index=False)
            print('Output file saved to: ' + output_file)
            
        elif method == 'tSNE':
            if n_dim > 3 or n_dim<2:
                print('tSNE method can reduce to 2 or 3 dimensions only\nExiting...')
                sys.exit(1)
            print('Starting tSNE method, reducing to n= {} dimensions'.format(self.n_dim))
            input_data = pd.read_csv(self.input_file)
            t0 = time.time()
            vec_embedded = TSNE(n_components=self.n_dim, init = 'pca', random_state =0).fit_transform(input_data)
            print('Embedding finished in {:.3} minutes'.format((time.time()-t0)/60))
            pd.DataFrame(vec_embedded).to_csv(self.output_file, index=False)
            print('Output file saved to: ' + output_file)
            
        else:
            print(method + 'is unfamiliar, current implemented methods: "PCA", "tSNE"\nExiting...')
            sys.exit(1)
            