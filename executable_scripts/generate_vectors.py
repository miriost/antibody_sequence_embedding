#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 10:31:42 2018

@author: miri-o
"""
import pandas as pd
import numpy as np
import os
import sys, argparse
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()).split('antibody_sequence_embedding')[0])            
from antibody_sequence_embedding import sequence_modeling


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', 
                        help='a *.csv file containing the JUNC_AA column')
    parser.add_argument('-o', '--odir',
                        help='output root directory', default = os.path.join(os.pardir, os.path.pardir))
    parser.add_argument('-c', '--column', 
                        help = 'column name to convert to vectors, default:"JUNC_AA"', default = "JUNC_AA")
    parser.add_argument('model', 
                        help='a saved word embedding model file')
    parser.add_argument('--size', help = "Vector size", default = 100)

    args = parser.parse_args()

    if not os.path.isfile(args.infile) or args.infile[:-4] == '.csv':
        print('Feature file ({}) error! Make sure the file exists and it is *.csv file.\nExiting...'.format(args.infile))
        sys.exit(1)
    print('Input file for embedding: ',args.infile, '\nModel: ', args.model)
 
    infile = pd.read_csv(args.infile, sep = '\t')
    filename = os.path.split(args.infile)[-1].split(os.path.extsep)[0]
    
    #load saved model
    model = sequence_modeling.load_protvec(args.model)
    modelname = os.path.split(args.model)[-1].split('.model')[0]
    
    # generate a vector for each junction
    data_len = len(infile)
    print('Data length: ' + str(data_len))
    W2V_vectors = np.zeros((data_len,int(args.size)))
    to_drop = []
    for i in range(data_len) :
        word = infile[args.column].iloc[i]
        if i%100000==0: 
            print(str(i) + ': ' + word)
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
    
    #save to files:

    path_filtered_files = os.path.join(args.odir, 'filtered_data_sets')
    if not os.path.exists(path_filtered_files):
        os.mkdir(path_filtered_files)
        
    infile_path = os.path.join(path_filtered_files, filename+'_'+modelname+'_FILTERED_DATA.tab')
    infile.to_csv(infile_path, sep='\t')
    print('Data file saved: ' + os.path.abspath(infile_path))
    
    path_vectors = os.path.join(args.odir, 'vectors')
    if not os.path.exists(path_vectors):
        os.mkdir(path_vectors)
        
    df_path = os.path.join(path_vectors, filename+'_'+modelname+ '_VECTORS.csv')
    df.to_csv(df_path, index = False)
    print('Vectors file saved: ' + os.path.abspath(df_path))

    
if __name__ == "__main__":
   main(sys.argv[1:])   
   
  
