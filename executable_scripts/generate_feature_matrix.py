#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 11:16:12 2018

@author: miri-o


"""


import sys, argparse
sys.path.insert(0, "/media/miri-o/Documents")
from miris_tools.sequence_modeling import generate_corpusfile

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('clustering_labels', help='a file containing the clustering labels (index: cluster)')
    parser.add_argument('classes', help='The correct class for each')
    parser.add_argument('-TH','--threshold', type=float, help = 'Threshold for filtering (Features where the precentage of observations from one subject > TH will be filtered)')
    
    args = parser.parse_args()
    print(args.clustering_labels, args.classes, str(args.threshold))

   
if __name__ == "__main__":
   main(sys.argv[1:])   
   
   
   
   
   
   # insert the clusters to data file         
        datafile = pd.read_csv(path+'HCV_data_for_PCA_20180529.csv', delimiter='\t')
        datafile['original index'] = fullfile.index
        datafile['cluster'] = [inv_labels[i] for i in datafile['original index']]
        log_file.close()
        
        
        # Build a feature table where each column is a cluster and each raw is a subject, 
        # count the number of sequnces of each subject in each cluster

        features_table = pd.DataFrame(0, index=pd.unique(datafile['SUBJECT']), columns=labels)
        for index, row in datafile.iterrows():
            features_table.loc[row.SUBJECT, row.cluster] +=1
        
        # Normlize by raw
        normlized_features_table = features_table.div(features_table.sum(axis=1), axis=0)

        fig, ax = plt.subplots(figsize=(15,6)) 
        sns.heatmap(normlized_features_table, cmap='viridis')
        
        ### Additional filtering - filter clusters with low subject diversity, i.e. X% of cell originated from one subject
        all_features_sum = normlized_features_table.sum(axis=0)
        all_features_max = normlized_features_table.max(axis=0)
        max_feature_precentage = all_features_max*100/all_features_sum
        max_feature_precentage = pd.DataFrame(max_feature_precentage, columns=['Precentage'])
        cut_off_TH = 95
        to_drop = pd.DataFrame([(center, value) for (center, value) in zip(max_feature_precentage.index, max_feature_precentage.Precentage) if value>cut_off_TH], columns= ['Cluster', 'Precentage_of_top_feature'])
        print(to_drop)
        
        # droping the above centers
        filtered_feature_table = normlized_features_table.drop(labels=to_drop.Cluster, axis = 1)
        print('Filtering {} columns where maximal feature precentage exceeds allowed TH, Centers: {}, Precentages: {}'.
              format(len(to_drop),list(to_drop.Cluster), list(to_drop.Precentage_of_top_feature)))
