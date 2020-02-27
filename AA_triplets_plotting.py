#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 13:13:40 2019

@author: miri-o
"""



import pandas as pd
import matplotlib.pyplot as plt
import os
from miris_tools import kmeans
        
props = ['CDR3_AA_GRAVY', 'CDR3_AA_BULK',
       'CDR3_AA_ALIPHATIC', 'CDR3_AA_POLARITY', 'CDR3_AA_CHARGE',
       'CDR3_AA_BASIC', 'CDR3_AA_ACIDIC', 'CDR3_AA_AROMATIC']

input_path = '/media/miri-o/Documents/AA_triplets_with_embedding_and_clusters.csv'
in_file = pd.read_csv(input_path)
data = pd.DataFrame(in_file)
output_dir = '/media/miri-o/Documents/results/New_results_dec5_2019'
# Basic plot of the emnedding
if False:
    fig = plt.figure(figsize=(8,8))
    plt.scatter(data['dim1'], data['dim2'], s=2)
    plt.title('3-gram embedding to 2-dimensional space', fontsize = 14)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(os.path.join(output_dir, '3_grams_without_coloring.png'), bbox_inches='tight')
    plt.close(fig)

#plot by all point with the property as color


for i in range(len(props)):
    
    output_filename = os.path.join(output_dir, props[i][8:]+'_property')
    kmeans.plot_embedding_with_properties(data['dim1'], data['dim2'], data[props[i]].tolist(),  filename = output_filename)
    
    
