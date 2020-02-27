#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 16:00:35 2019

@author: miri-o
"""

#import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
#import scipy
import numpy as np
#import sklearn
from sklearn.manifold import TSNE
import statistics
#import itertools
import sys, getopt
sys.path.insert(0, "/media/miri-o/Documents/")
sys.path.insert(0, "/media/miri-o/Documents/miris_tools")
import time
import biovec
from miris_tools import kmeans
from scipy.spatial import distance_matrix
#import statistics
from collections import Counter
import os

def draw_plot(data, edge_color, fill_color, median_color, blabels = None):
    medianprops = dict(linewidth=2.5, color='red') 
    bp = ax.boxplot(data, patch_artist=True, labels = blabels, medianprops = medianprops, showmeans = True, widths = 0.6)
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)

    for patch in bp['boxes']:
        patch.set(facecolor=fill_color) 

def imap(function, *iterables):
    # imap(pow, (2,3,10), (5,2,3)) --> 32 9 1000
    iterables = map(iter, iterables)
    while True:
        args = [next(it) for it in iterables]
        if function is None:
            yield tuple(args)
        else:
            yield function(*args)        

def build_clust_logo(data_frame, number_of_clusters):
    # for each cluster, build a vector with size 20, with the distribution of each amino acid within the logo
    index = range(number_of_clusters)
    columns = ['length', 'logo']
    df_ = pd.DataFrame(index=index, columns=columns)
    df_ = df_.fillna(0) # with 0s rather than NaNs   
    by_cluster = data_frame.groupby('cluster')
    df_['length'] = by_cluster.count()['Ngram']
    for name, clusterrr in by_cluster:
        Ngrams = ''.join(list(clusterrr.Ngram))
        counter = Counter(Ngrams)
        df_.loc[name, ['logo']] = dict(counter)

    print(df_)
        
## load model:
model_path = '/media/miri-o/Documents/trained_models_NEW/celiac_model_Jan19_2019.model'
CDR3_cropped_model = biovec.models.load_protvec(model_path)

# load n_gram list with properties (calculated in R alakazam)

      
    

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


# read n-grams with properties (created by alakazm) to data frame

ngrams_properties = pd.read_csv("/media/miri-o/Documents/prot_ngrams_properties.csv")
prot_ngram_vecs = {}
prot_ngram_vecs = prot_ngram_vecs.fromkeys(ngrams_properties.Ngram, 0)

# crete a vector for each sequence
for ngram in ngrams_properties.Ngram:
    try:
        prot_ngram_vecs[ngram] = list(CDR3_cropped_model.to_vecs(ngram)[0])
    except:
        print('Could not convert ngram: ', ngram) 
prot_ngrams_df = pd.DataFrame(prot_ngram_vecs)
prot_ngrams_array = np.transpose(prot_ngrams_df.values)    

print('Emedding finished, reducing dimensions using TSNE')
t0 = time.time()
vec_embedded_8000 = TSNE(n_components=2, init = 'pca', random_state =0).fit_transform(prot_ngrams_array)

print('Dimension reduction finished in {:.3} minutes'.format((time.time()-t0)/60))

plt.scatter(vec_embedded_8000[:,0], vec_embedded_8000[:,1], s=2)
#create a grid to sperate the points   
plt.show()

# add embedding to the ngrams data frame
ngrams_properties['dim1'] = vec_embedded_8000[:,0]
ngrams_properties['dim2'] = vec_embedded_8000[:,1]


props = ['CDR3_AA_GRAVY', 'CDR3_AA_BULK',
       'CDR3_AA_ALIPHATIC', 'CDR3_AA_POLARITY', 'CDR3_AA_CHARGE',
       'CDR3_AA_BASIC', 'CDR3_AA_ACIDIC', 'CDR3_AA_AROMATIC']

#plot each plot with a specific color
fig = plt.figure(figsize=(14, 10))
for i in range(len(props)):
    fig.add_subplot(2,4,i+1)
    kmeans.plot_embedding_with_properties(ngrams_properties['dim1'], ngrams_properties['dim2'], ngrams_properties[props[i]].tolist(), title=props[i][8:])


# ~~ CLUSTERING:

n=20 #number of clusters in each layer
depth = 2 #depth of clustering 
debug_mode = True
clust = kmeans.unsupervised_clustering(n, depth, debug_mode)
clust.fit(pd.DataFrame(vec_embedded_8000))
clust.visualize() 

ngrams_properties['cluster'] = [clust.labels_[i] for i in range(len(ngrams_properties))] 
#save to csv


#output_path = '/media/miri-o/Documents/AA_triplets_with_embedding_and_clusters.csv'
#ngrams_properties.to_csv(output_path)

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
output_dir = '/media/miri-o/Documents/results/New_results_dec5_2019/'

dist_means = []
clust_means = []

for prop in props:
    prop_df = pd.DataFrame(ngrams_properties, columns=[prop])
    dist_prop = distance_matrix(prop_df.values, prop_df.values)
    dist_means.append(np.mean(dist_prop, axis = 1)) # a matrix [num_props x num_points], for each property, the avreage distance of each points from all the other points

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
    
    clust_means.append(clust_distance)# a matrix [num_props x num_points], for each property, the avreage distance of each points from all the other points in it's cluster
    
for prop in range(len(props)):  
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.subplots_adjust(left=0.075, right=0.95, top=1, bottom=0.25)
    #pos = np.array(range(len(bulk))) + 1
    #bp = ax.boxplot(bulk, sym='k+', positions=pos,
    #                notch=1, bootstrap=5000, widths = 0.5, labels = ['within cluster', 'all data'])    
    
    draw_plot([clust_means[prop], dist_means[prop]], 'black', 'lightpink','red', blabels = ['within cluster', 'all data'])
    plt.title(props[prop])
    plt.xlim(0.5, 2.5)
    plt.savefig(os.path.join(output_dir, props[prop]+ '_variance.png'), bbox_inches='tight')
    plt.show()

# plot one cluster
j=248
fig, ax = plt.subplots(figsize=(10, 7))
clust1 = ngrams_properties.iloc[list(clust.clusters_[j])]
ax.scatter(clust1['dim1'], clust1['dim2'], c=clust1['CDR3_AA_AROMATIC'])

                   
for i,type in enumerate(clust1['Ngram'].values):
    x = clust1['dim1'].values[i]
    y = clust1['dim2'].values[i]
    c = clust1['CDR3_AA_AROMATIC'].values[i]

    plt.text(x+.1, y, type, fontsize=9)
    
plt.title('Cluster #{} example, colored by aromatic property'.format(j))
plt.show()


