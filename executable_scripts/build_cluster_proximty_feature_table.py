# -*- coding: utf-8 -*-
#!/bin/sh
''''exec python -u -- "$0" ${1+"$@"} # '''

"""
Created on Thu Apr 23 10:18:13 2020

@author: mirio
"""
"""
Cluster proximity to feature table

Feture generation script 
================
Input:
Feature list - a file containing the relevant clusters (max distance + center)
data file
vectors file

Output:
    Feature file, where each row is a subject, each column is a sequence, each cell is the normlized frequency of sequences in each feature
    
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from scipy.spatial.distance import euclidean
from scipy.spatial import distance_matrix
import sys, argparse
import os
from datetime import datetime
import time

def main():
    
    if len(sys.argv) <= 1:
        sys.argv.extend(" -f ../../cross_validation/try1.csv -d ../../filtered_data_sets/CDR3_from_celiac_trim_3_4_with_labels_unique_sequences_Celiac_model_April_2020_FILTERED_DATA_1K_per_subject.csv -v ../../vectors/CDR3_from_celiac_trim_3_4_with_labels_unique_sequences_Celiac_model_April_2020_VECTORS_1K_per_subject.csv -of ../../cross_validation/ -od try1K_TRAIN_0".split())
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--features_list',
                        help='feature list file, contains the list of relevent features, including feature center and maximal distance from it')
    parser.add_argument('-d', '--data_file_path',
                        help='the filtered data file path')
    parser.add_argument('-v', '--vectors_file_path',
                        help='the vectors file path')
    parser.add_argument('-of', '--output_folder_path',
                        help='Output folder for the feature table')
    parser.add_argument('-od', '--output_description',
                        help='description to use inside output file names')
    parser.add_argument('-s', '--subject_col_name',
                        help='subject column name in data file, default "FILENAME"', default='FILENAME', type=str)
    parser.add_argument('-l', '--labels_col_name',
                        help='labels column name in data file, default "labels"', default='labels', type=str)
    args = parser.parse_args()
    
    if not(os.path.isfile(args.features_list)):
        print('feature list file error, make sure file path exists\nExiting...')
        sys.exit(1)
                
    if not(os.path.isfile(args.data_file_path)):
        print('feature file error, make sure file path exists\nExiting...')
        sys.exit(1)
        
            
    if not(os.path.isfile(args.vectors_file_path)):
        print('vectors file error, make sure file path exists\nExiting...')
        sys.exit(1) 
        
    
    # load files    
    feature_list = pd.read_csv(args.features_list, index_col=0)
    data_file = pd.read_csv(args.data_file_path)
    vectors_file = pd.read_csv(args.vectors_file_path)

    if not args.labels_col_name in data_file.columns:
        print(f'label "{args.labels_col_name}" column name doesnt exist in data file.\nExiting...')
        sys.exit(1)
        
    if not args.subject_col_name in data_file.columns:
        print(f'"{args.subject_col_name}" column name doesnt exist in data file.\nExiting...')
        sys.exit(1)
    
    if not 'feature_index' in feature_list.columns:
        print(f'"feature list file error, no "feature index" column. please check.\n exiting...')
        sys.exit(1)
    else:
        print(f'feature indexes: {feature_list.index}')
    features_table = pd.DataFrame(0, index=pd.unique(data_file[args.subject_col_name]), columns=feature_list['feature_index']) #define an empty matrix, each raw is a subject, each column is a feature (cluster)
                            
    by_subject = data_file.groupby(args.subject_col_name)
    sub_num = 0
    
    for subject, frame in by_subject: # for each subject
        sub_num += 1
        print("------------------------")
        print(f"{str(datetime.now())}: Analysing {subject!r} #{sub_num!r}")
        for vector_index, row in frame.iterrows(): #for each vector in that subject
            #print(f"{str(datetime.now())}: Analysing {vector_index!r} vector index")
            sum_iloc = 0.0
            cnt_iloc = 0
            sum_euclidean = 0.0
            cnt_eculidean = 0
            start_time_others = time.time()
            #features_count = 0
            multiple_entries = 0

            vector_u = vectors_file.iloc[vector_index, :] # vector in data file
            if True:
                # pavel new
                features = feature_list.iloc[:, -100:]
                distances = distance_matrix(features, np.array(vector_u, ndmin=2))
                distances = distances.reshape((len(features), ))
                max_distance = feature_list.loc[:, 'max_distance']
                distance_close_enough_vec = distances <= max_distance
                # TODO: where to increment the counters?
                features_count = np.sum(distance_close_enough_vec)
                if features_count>1:
                    multiple_entries +=1
                add_feature_index = np.where(distance_close_enough_vec==True)
                features_table.loc[subject, feature_list.loc[add_feature_index, 'feature_index']] += 1

            if False: # thecode before
                for feature_index in feature_list.index: #check distances of each vector from all features
                    tic = time.time()
                    vector_v = feature_list.iloc[feature_index, -100:] #center vector is the last 100 vectors
                    sum_iloc += time.time() - tic
                    cnt_iloc +=1

                    tic = time.time()
                    distance = euclidean(vector_u, vector_v)
                    sum_euclidean += time.time() - tic
                    cnt_eculidean +=1

                    if distance <= feature_list.loc[feature_index, 'max_distance']:
                        # print(f'feature {feature} answers condition')
                        features_table.loc[subject, feature_list.loc[feature_index, 'feature_index']] += 1
                        features_count+=1

                print("first iloc time = {}ms cnt={}\t eculedian time={}ms\t all={}".format(
                   1000 * sum_iloc/ cnt_iloc, cnt_iloc, 1000 * sum_euclidean / cnt_eculidean, time.time()-start_time_others))
        # print(f'===> A total of {features_count} answered the conditions, out of {len(frame)} raws')
               
        # Normlize by raw
 
    normlized_features_table = features_table.div(features_table.sum(axis=1), axis=0)
    normlized_features_table.to_csv(os.path.join(args.output_folder_path, args.output_description + '_feature_table.csv'))
      
    print('file saved to ', os.path.join(args.output_folder_path, args.output_description + '_feature_table.csv'))

def einsum(v, u):
   z = v - u
   return np.sqrt(np.einsum('i,i->', z, z))

def bare_numpy(v, u):
   return np.sqrt(np.sum((v - u) ** 2))
    
if __name__ == '__main__':
    main()



# ====
# build a feature table - the rows of the table is the subject, the columns are the clusters - cluster name is according to the index of the cell from which the cluster orignated.
if False:
    data_file_path = r'C:\Users\mirio\research\filtered_data_sets\CDR3_from_celiac_trim_3_4_with_labels_unique_sequences_Celiac_model_April_2020_FILTERED_DATA_10K_per_subject.csv'
    datafile = pd.read_csv(data_file_path)
    clusters = np.nonzero(~np.isnan(selected_feature_indexes))[0]
    features_table = pd.DataFrame(0, index=pd.unique(datafile['FILENAME']), columns=clusters)
    indexes_for_feature_table = np.nonzero(~np.isnan(neigbors_feature_index))[0]
    for element in indexes_for_feature_table:
        features_table.loc[datafile.FILENAME[element], neigbors_feature_index[element]] +=1
    # Normlize by raw
    normlized_features_table = features_table.div(features_table.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(15,6)) 
    sns.heatmap(normlized_features_table, cmap='viridis')
    ax.set_title('Normzliazed feature table, Celiac, 10K per subject', fontsize=14)
    normlized_features_table.to_csv(r'C:\Users\mirio\research\feature_tables\Celiac_method2_10K_per_subject_feature_table.csv', index_label = 'FILENAME')

    
    ##=== create a file with feature table indexes on unseen data
    #
    feature_table = pd.read_csv(r'C:\Users\mirio\research\feature_tables\Celiac_method2_10K_per_subject_feature_table.csv', index_col = 'FILENAME')
    features_indexes = feature_table.columns
    num_of_features = len(features_indexes)
    print(f'Number of features: {num_of_features}')
    
    dist_file = pd.read_csv(r'C:\Users\mirio\research\10K_test\features_distances_554.csv', index_col = 0)
    vec_file = pd.read_csv(r'C:\Users\mirio\research\10K_test\features_vectors_554.csv', index_col = 0)
    
    verify_file_data = pd.read_csv(r'C:\Users\mirio\research\filtered_data_sets\CDR3_from_celiac_trim_3_4_with_labels_unique_sequences_Celiac_model_April_2020_VERIFY_10K_per_subject.csv')
    number_of_subjects = len(np.unique(verify_file_data.FILENAME))
    verify_file_vectors = pd.read_csv(r'C:\Users\mirio\research\vectors\CDR3_from_celiac_trim_3_4_with_labels_unique_sequences_Celiac_model_April_2020_VECTORS_VERIFY_10K_per_subject.csv')
    
    original_vec_file = pd.read_csv(r'C:\Users\mirio\research\vectors\CDR3_from_celiac_trim_3_4_with_labels_unique_sequences_Celiac_model_April_2020_VECTORS_10K_per_subject.csv')
    
    validation_feature_table = pd.DataFrame(index = np.unique(verify_file_data.FILENAME), columns = features_indexes)
    validation_feature_table = validation_feature_table.fillna(0)
                            
    by_subject = verify_file_data.groupby('FILENAME')
    sub_num = 0
    
    for subject, frame in by_subject:
        sub_num += 1
        print("------------------------")
        print(str(datetime.now()) + f" : Analysing {subject!r} #{sub_num!r}")
        for index, row in frame.iterrows():
            features_count = 0
            vector_u = verify_file_vectors.iloc[index, :]
            for feature in features_indexes:
                #calculate the distance of the vector from each one of the features, and compare it to the distances, if it is smaller than the maximal distance, add 1 to the feature table
                vector_v = original_vec_file.iloc[int(feature), :]
                distance = euclidean(vector_u, vector_v)
                if distance <= max(dist_file.loc[int(feature), :]):
                    print(f'feature {feature} answers condition')
                    validation_feature_table.loc[subject, feature] +=1
                    features_count+=1
        print(f'===> A total of {features_count} answered the conditions, out of {len(frame)} rows')
    # Normlize by raw
    normlized_features_table = validation_feature_table.div(validation_feature_table.sum(axis=1), axis=0)
    normlized_features_table.to_csv(r'C:\Users\mirio\research\feature_tables\Celiac_method2_10K_per_subject_VERIFY_feature_table.csv', index_label = 'FILENAME')
