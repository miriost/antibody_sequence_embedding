# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 16:32:22 2020

@author: mirio
"""

#reduce data by adding a count for unique sequence (after trimming)
import pandas as pd
from collections import Counter
import pprofile
import numpy as np
	
from datetime import datetime

Input_path = r'C:\Users\mirio\research\data\CDR3_from_celiac_trim_3_4_with_labels.csv'
Input_file = pd.read_csv(Input_path)
Input_file = Input_file.drop(['Unnamed: 0'], axis = 1)
print('Original file length: {}'.format(len(Input_file)))
Subject_list = Input_file['FILENAME'].unique()
#print subject summary
Counter_before = Counter(Input_file['FILENAME'])

df_columns = list(Input_file.columns)
df_columns.append('UNIQCOUNT')
#unique_df = pd.DataFrame(np.nan, index=range(0,len(Input_file)), columns = df_columns)
row_index = 0
sub_num = 1
list_test = [np.nan]*len(Input_file)

profiler = pprofile.Profile()
#with profiler:
by_subject = Input_file.groupby(['FILENAME'])
for subject, frame in by_subject:
    print(f"Analysing {subject!r} {sub_num!r}")
    print("------------------------")
    dateTimeObj = datetime.now()
    timeObj = dateTimeObj.time()
    print(timeObj)
    print('number of sequences before: ', len(frame), end="\n\n")
    c=0
    by_sequence = frame.groupby('JUNC_AA')
    for sequence, data in by_sequence:
        one_row = data.head(1)
        one_row = one_row.assign(UNIQCOUNT=len(data))
        #unique_df.iloc[row_index] = one_row.values[0]
        list_test[row_index] = one_row.values[0]
        row_index +=1
        c+=1
        
    sub_num+=1                
    print('number of sequences after: ', c, end="\n\n")

print('ALL DONE!!, converting to data frame')
unique_df = pd.DataFrame(list_test[:row_index], columns = df_columns)   
unique_df.to_csv(r'C:\Users\mirio\research\data\CDR3_from_celiac_trim_3_4_with_labels_unique_sequences.csv')
#profiler.print_stats()        
        
#    sub_df = Input_file[Input_file['FILENAME']==subject]
#    print('Subject #{}, filename:{}, length before: {}, unique sequences: {}'.format(i, subject, len(sub_df), len(sub_df['JUNC_AA'].unique())))
    
    
    