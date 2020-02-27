#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 16:00:35 2019

@author: miri-o
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import statistics
import itertools
import sys, getopt
sys.path.insert(0, "/media/miri-o/Documents/")
sys.path.insert(0, "/media/miri-o/Documents/miris_tools")
import time
import biovec
from miris_tools import kmeans
from scipy.spatial import distance_matrix
import statistics


def draw_plot(data, edge_color, fill_color, median_color, blabels = None):
    medianprops = dict(linewidth=2.5, color='red') 
    bp = ax.boxplot(data, patch_artist=True, labels = blabels, medianprops = medianprops, showmeans = True, widths = 0.6)
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)

    for patch in bp['boxes']:
        patch.set(facecolor=fill_color) 
      
    

# take 8000 3-grams and map them the same way

n = 3
#prot_ngrams = [list(n) for n in itertools.product('ACDEFGHIKLMNPQRSTVWY', repeat = n)]
#prot_ngrams = [''.join(ngram) for ngram in prot_ngrams]
#prot_ngram_vecs = {}
#prot_ngram_vecs = prot_ngram_vecs.fromkeys(prot_ngrams, 0)
#
#for ngram in prot_ngrams:
#    try:
#        prot_ngram_vecs[ngram] = list(CDR3_cropped_model.to_vecs(ngram)[0])
#    except:
#        print('Could not convert ngram: ', ngram) 
#prot_ngrams_df = pd.DataFrame(prot_ngram_vecs)
#prot_ngrams_array = np.transpose(prot_ngrams_df.values)    


ngrams_properties = pd.read_csv("/media/miri-o/Documents/AA_triplets_with_embedding_and_clusters.csv")
props = ['CDR3_AA_GRAVY', 'CDR3_AA_BULK',
       'CDR3_AA_ALIPHATIC', 'CDR3_AA_POLARITY', 'CDR3_AA_CHARGE',
       'CDR3_AA_BASIC', 'CDR3_AA_ACIDIC', 'CDR3_AA_AROMATIC']


# calculating the variance of a property within each cluster vs. each general variance:

for prop in props:
    num_clusters = 0
    sample_size = 0
    genvar = statistics.variance(ngrams_properties[prop])
    print('property name: {}, General variance: {:.3f}'.format(prop, genvar))
    var_sum = 0
    #pooled variance
    for i in range(clust.n_clusters_):
        n_i = len(clust.clusters_[i])
        if n_i > 1:
            
            num_clusters +=1
            sample_size += n_i
            var_sum += (n_i-1)*statistics.variance(ngrams_properties[prop][clust.clusters_[i]])
    invar = var_sum/(sample_size-num_clusters)
    print('In-cluster pooled variance: {:.3f}, ratio: {:.3f}'.format(invar, genvar/invar))
        

# calculating the distance matrix for all points:
xy_couples = pd.DataFrame(ngrams_properties, columns=['dim1','dim2'])
dist_mat_all = distance_matrix(xy_couples.values, xy_couples.values)

#property distance for each property vs. the distance within each cluster

dist_means = []
clust_means = []

for prop in props:
    prop_df = pd.DataFrame(ngrams_properties, columns=[prop])
    dist_prop = distance_matrix(prop_df.values, prop_df.values)
    dist_means.append(np.mean(dist_prop, axis = 1))

#fig, ax = plt.subplots(figsize=(15, 10))
#pos = np.array(range(len(dist_means))) + 1
#bp = ax.boxplot(dist_means, sym='k+', positions=pos,
#                notch=1, bootstrap=5000)
#ax.set_xticklabels(props, rotation=45, fontsize=8) 
        
#distances of properties within each cluster 
    
    clust_distance = []
    for i in clust.clusters_:
        clust_df =  pd.DataFrame(ngrams_properties[prop].iloc[list(clust.clusters_[i])])
        dist_clust = distance_matrix(clust_df.values, clust_df.values)
        clust_distance.extend(np.mean(dist_clust, axis=1))
    
    clust_means.append(clust_distance)
    
for prop in range(len(props)):  
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.subplots_adjust(left=0.075, right=0.95, top=1, bottom=0.25)
    #pos = np.array(range(len(bulk))) + 1
    #bp = ax.boxplot(bulk, sym='k+', positions=pos,
    #                notch=1, bootstrap=5000, widths = 0.5, labels = ['within cluster', 'all data'])    
    
    draw_plot([clust_means[prop], dist_means[prop]], 'black', 'lightpink','red', blabels = ['within cluster', 'all data'])
    plt.title(props[prop])
    plt.xlim(0.5, 2.5)
    
    plt.show()

#
#ax.set_xlabel('treatment')
#ax.set_ylabel('response')
#plt.setp(bp['whiskers'], color='k', linestyle='-')
#plt.setp(bp['fliers'], markersize=3.0)
#plt.show()