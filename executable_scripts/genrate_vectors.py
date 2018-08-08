#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 10:31:42 2018

@author: miri-o
"""

import sys, argparse
sys.path.insert(0, "/media/miri-o/Documents")
from miris_tools import sequence_modeling
import pandas as pd
import numpy as np

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', help='a *.csv file containing the JUNC_AA column')
    parser.add_argument('model', help='a saved word embedding model file')
        
    args = parser.parse_args()
    print('Input file for embedding: ',args.infile, '\nModel: ', args.model)
 
    infile = pd.read_csv(args.infile)
    filename = args.infile.split('/')[-1]
    
    #load saved model
    model = sequence_modeling.load_protvec(args.model)
    modelname = args.model.split('/')[-1]
    
    # generate a vector for each junction
    data_len = len(infile)
    W2V_vectors = np.zeros((data_len,100))
    to_drop = []
    for i in range(data_len) :
        word = infile['JUNC_AA'].iloc[i] 
        # Check for errors (if some sequence doesn't appear in the model)
        try:
            W2V_vectors[i] = list(model.to_vecs(word)[0])
        except:
            W2V_vectors[i] = np.nan
            to_drop.append(i)
            #print(str(i) + ' index not valid')
    print('{:.3}% of data not transformed'.format((100*len(to_drop)/data_len)))
    
    # drop the un translated rows from the file
    infile = infile.drop(infile.index[to_drop])
    df = pd.DataFrame(W2V_vectors)
    df = df.drop(df.index[to_drop])
    #print(len(infile), len(df))
    
    #save to fileS:
    path_filtered_files = '/media/miri-o/Documents/Immune2vec/filtered_data_sets/'
    path_vectors = '/media/miri-o/Documents/Immune2vec/vectors/'
    infile.to_csv(path_filtered_files+filename[:-4]+'_'+modelname[:-6]+'_FILTERED_DATA.csv')
    print('Data file saved: ' + path_filtered_files+filename[:-4]+'_'+modelname[:-6]+'_FILTERED_DATA.csv')
    df.to_csv(path_vectors+filename[:-4]+'_'+modelname[:-6]+'_VECTORS.csv', index = False)
    print('Vectors file saved: ' + path_vectors+filename[:-4]+'_'+modelname[:-6]+'_VECTORS.csv')

    
if __name__ == "__main__":
   main(sys.argv[1:])   
   
  