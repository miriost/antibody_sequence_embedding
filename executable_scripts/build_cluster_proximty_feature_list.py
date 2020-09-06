# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 10:18:13 2020

@author: mirio
"""
"""
Cluster proximity to feature table

Feture selection
================
Start by feature selection according to the following parameters:
    1. min_number_of_subjects- how many subjects are in each cluster
    2. Significance - only clusters with higher score will be chosen abs(1-score) 
    * We will want to eliminate clusters that appeared already
    
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
import sys, argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--analysis_file_path',
                        help='the filtered data file path')
    parser.add_argument('-ds', '--distances_file_path',
                        help='the distances file path')
    # parser.add_argument('-nn', '--NN_file_path',
    #                     help='the NN file path')
    # parser.add_argument('-d', '--data_file_path',
    #                     help='the filtered data file path')
    parser.add_argument('-v', '--vectors_file_path',
                        help='the vectors file path')
    parser.add_argument('-of', '--output_folder_path',
                        help='Output folder for the 3 output files - Nearest neighbors file, destances file, and results anaylsis file')
    parser.add_argument('-od', '--output_description',
                        help='description to use inside output file names')
    parser.add_argument('-S','--significance', help = 'minimal significance score for cluster selection', default=0.6)
    parser.add_argument('-m','--min_subjects', help = 'minimal significance score for cluster selection', default=10, type=int)
    args = parser.parse_args()
    
    if not(os.path.isfile(args.analysis_file_path)):
        print('analysis file error, make sure file path exists\nExiting...')
        sys.exit(1)
        date
    if not(os.path.isfile(args.distances_file_path)):
        print('distances file error, make sure file path exists\nExiting...')
        sys.exit(1)
        
    # if not(os.path.isfile(args.NN_file_path)):
    #     print('NN file error, make sure file path exists\nExiting...')
    #     sys.exit(1)
        
    # if not(os.path.isfile(args.data_file_path)):
    #     print('feature file error, make sure file path exists\nExiting...')
    #     sys.exit(1)
        
            
    if not(os.path.isfile(args.vectors_file_path)):
        print('vectors file error, make sure file path exists\nExiting...')
        sys.exit(1) 
        
    
    # load files    
    inputfile = pd.read_csv(args.analysis_file_path)
   # NN_file = pd.read_csv(args.NN_file_path)
    #data_file = pd.read_csv(args.data_file_path)
    vectors_file = pd.read_csv(args.vectors_file_path)
    distance_file = pd.read_csv(args.distances_file_path)
    
    #load parameters
    min_number_of_subjects = args.min_subjects
    min_significance = args.significance

    #calculate significance score    
    inputfile['SIGNIFICANCE'] = [max(a,1-a) for a in inputfile['HC']]
    sorted_df = inputfile.sort_values(by=['SIGNIFICANCE'], ascending=False)
    
    #processing
    
    number_of_features_neto = 0
    number_of_features_bruto = 0
    selected_feature_indexes = [np.nan] * len(inputfile) # A list with the same length as the data, nan on all cells except the chosen features, which will have 1
    neigbors_feature_index = [np.nan] * len(inputfile) # A list with the same length as the data, nan on all cells except the chosen points, which will have the index of the original cell which created the cluster
    for idx, val in sorted_df.iterrows():
        if val['SIGNIFICANCE'] > min_significance and val['how_many_subjects'] > min_number_of_subjects:
            number_of_features_bruto +=1
            if np.isnan(neigbors_feature_index[idx]):
                number_of_features_neto+=1
                selected_feature_indexes[idx] = 1
                for neighbor in map(int, val['neighbors'][1:-1].split(',')): 
                    neigbors_feature_index[neighbor] = idx

    print('Summary\n=========\n')                     
    print('Significant score: '+str(min_significance))           
    print('Number of features that meet the TH criteria:' + str(number_of_features_bruto))
    print('Number of features after filtration of those appeared as neighbors: ' + str(number_of_features_neto))
    
    # ====
    # build a feature list file, each raw contains feature center, and maximal radius
    c = ['feature_index','max_distance']
    c.extend(vectors_file.columns)
    print(c)
    selected_features = np.nonzero(~np.isnan(selected_feature_indexes))[0]
    print('selected features: ', selected_features)
    if len(selected_features) != number_of_features_neto:
        print('Error! selected features number mismatch!!')
    else:
        print(f'Beginning builiding feature list with {number_of_features_neto} features')
    
    features_df = pd.DataFrame(0, index=range(0,number_of_features_neto), columns=c)
    for index, value in enumerate(selected_features):
        local_max_distance = max(distance_file.iloc[value])
        tmp = [value, local_max_distance]
        tmp.extend(list(vectors_file.iloc[value]))
        features_df.iloc[index] = tmp
        # print(features_df.iloc[index])
        
    features_df.to_csv(os.path.join(args.output_folder_path, args.output_description + '.csv'))
    print('file saved to ', os.path.join(args.output_folder_path, args.output_description + '.csv'))

    
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
        print(f"Analysing {subject!r} #{sub_num!r}")
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
