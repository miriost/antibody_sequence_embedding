#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 15:38:44 2019

@author: miri-o
"""
import pandas as pd
from collections import Counter
from scipy.spatial import distance_matrix, distance
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import spatial
import numpy as np

def build_clust_logo(data_frame):
    # for each cluster, build a vector with size 20, with the distribution of each amino acid within the logo
    by_cluster = data_frame.groupby('cluster')
    index = by_cluster.groups.keys()
    columns = ['length', 'center_x', 'center_y']+list('LASGVERTDIKPNQFYMHCW')
    df_ = pd.DataFrame(index=index, columns=columns)
    df_ = df_.fillna(0) # with 0s rather than NaNs   
    df_['length'] = by_cluster.count()['Ngram']
    df_['center_x'] = by_cluster.mean()['dim1']
    df_['center_y'] = by_cluster.mean()['dim2']
    for name, clusterrr in by_cluster:
        c = Counter({'L':0,'A':0, 'S':0, 'G':0, 'V':0, 'E':0, 'R':0, 'T':0, 'D':0, 'I':0, 'K':0, 'P':0, 'N':0, 'Q':0, 'F':0, 'Y':0, 'M':0, 'H':0, 'C':0, 'W':0})
        for Ng in clusterrr.Ngram:
            #print('Adding Ngram: {}, to cluster number: {}'.format(Ng, name))
            c.update(Ng)
        df_.at[name, c.keys()] = [val/df_.at[name, 'length'] for val in c.values()]

    #print(df_)
    return(df_)
    
def clusterize_properties(df, props):
    by_cluster = df.groupby('cluster')
    index = by_cluster.groups.keys()
    df_ = pd.DataFrame(index=index, columns=props)
    for column in props:
        df_[column] = by_cluster.mean()[column]
    print(df_)
    return(df_)
    
    
def property_distance_vector(vec):
#    Input: property vector : [a_o, a_1, a_2..., a_n]
#    Output: p2 norm distance vector: [d_0-1, d_0-2, ..., d_0-n, d_1-2,..., d_1-n...], 
    l = len(vec)
    out_vec =[]
    for i in range(0, l):
        for j in range(i+1, l):
            out_vec.extend([np.abs(vec[i]-vec[j])])
            
    return(out_vec)
            
    
in_file = pd.read_csv('/media/miri-o/Documents/AA_triplets_with_embedding_and_clusters.csv')
data = pd.DataFrame(in_file, columns = ['Ngram', 'cluster', 'dim1', 'dim2'])

props = ['CDR3_AA_GRAVY', 'CDR3_AA_BULK',
       'CDR3_AA_ALIPHATIC', 'CDR3_AA_POLARITY', 'CDR3_AA_CHARGE', 'CDR3_AA_BASIC', 'CDR3_AA_ACIDIC', 'CDR3_AA_AROMATIC']

property_data = pd.DataFrame(in_file)
property_data = property_data.drop(['Ngram', 'dim1', 'dim2'], axis=1)
property_data_clusterized = clusterize_properties(property_data , props)

amino_acid_logo = build_clust_logo(data)
amino_acid_logo.to_csv('/media/miri-o/Documents/results/amino_acids_clusters_logo.csv')

amino_acid_logo_values = amino_acid_logo.drop(['length', 'center_x', 'center_y'], axis = 1)
# compute ditance matrix
logo_cluster_dist_mat = pd.DataFrame(distance_matrix(amino_acid_logo_values.values, amino_acid_logo_values.values))
plt.figure(figsize=(12, 12))
sns.heatmap(logo_cluster_dist_mat, cmap = "RdBu")

plt.show()




fig2 = plt.figure(figsize=(10,10))
ax2 = plt.scatter(data['dim1'], data['dim2'], s=3, marker = 'D')
ax2 = plt.scatter(amino_acid_logo['center_x'], amino_acid_logo['center_y'], s=3, marker = 'x', color = 'r')

amino_acid_coords = amino_acid_logo[['center_x', 'center_y']]
CM_cluster_dist_mat = pd.DataFrame(distance_matrix(amino_acid_coords.values, amino_acid_coords.values))
plt.figure(figsize=(12, 12))
sns.heatmap(CM_cluster_dist_mat, cmap = "RdBu")

plt.show()

CM_vector = spatial.distance.squareform(CM_cluster_dist_mat)
logo_vector = spatial.distance.squareform(logo_cluster_dist_mat)

fig3 = plt.figure(figsize=(10,10))
ax3 = plt.scatter(CM_vector, logo_vector, s=1.6)
print('length CM vector: {}'.format(len(CM_vector)))
plt.xlabel('Centeral mass distance')
plt.ylabel('AA frequency distance')
plt.show()

sns.set('paper', 'white', rc={'font.size': 10, 'axes.labelsize': 10, 'legend.fontsize': 8, 'axes.titlesize': 10,
                              'xtick.labelsize': 8,
                              'ytick.labelsize': 8, "pgf.rcfonts": False})
plt.rc('font', **{'family': 'serif', 'serif': ['Times']})
plt.rc('text', usetex=False)

data_frame_for_plotting = pd.DataFrame(CM_vector, columns = ['CM'])
data_frame_for_plotting['logo'] = logo_vector
sns.distplot(CM_vector)
sns.distplot(logo_vector)

for prop in props:
    prop_vector = property_distance_vector(property_data_clusterized[prop].values)
    data_frame_for_plotting[prop] = prop_vector

    
    fig4 = plt.figure(figsize=(8,8))
#    ax4 = plt.scatter(CM_vector, prop_vector, s=1.6, color = 'm')
    sns.jointplot(x = 'CM', y = prop, data = data_frame_for_plotting, kind = 'kde', cmap = "Reds")
    
    plt.xlabel('Centeral mass distance', fontsize=14)
    plt.ylabel('property distance between clusters', fontsize=14)
    plt.title(prop[8:], fontsize=20)
    plt.show()
    fig4.savefig('/media/miri-o/Documents/results/{}_CM_sns.pdf'.format(prop[8:]))
    
    fig5 = plt.figure(figsize=(8,8))
#    ax5 = plt.scatter(logo_vector, prop_vector, s=1.6, color = 'b')
    sns.jointplot(x = 'logo', y = prop, data = data_frame_for_plotting, kind = 'kde', cmap = "Blues")
    plt.xlabel('logo distance', fontsize=14)
    plt.ylabel('property distance between clusters', fontsize=14)
    plt.title(prop[8:], fontsize=20)
    plt.show()
    fig5.savefig('/media/miri-o/Documents/results/{}_logo_sns.pdf'.format(prop[8:]))

