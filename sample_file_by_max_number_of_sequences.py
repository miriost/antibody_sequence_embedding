# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 14:57:21 2020

@author: mirio
"""

# sample file 

import pandas as pd
max_num_smaples_per_subject = 10000

Input_filtered_data_file_path = r'C:\Users\mirio\research\filtered_data_sets\CDR3_from_celiac_trim_3_4_with_labels_unique_sequences_Celiac_model_April_2020_FILTERED_DATA.csv'
Output_filtered_data_file_path = r'C:\Users\mirio\research\filtered_data_sets\CDR3_from_celiac_trim_3_4_with_labels_unique_sequences_Celiac_model_April_2020_VERIFY_10K_per_subject.csv'
Input_vector_file_path = r'C:\Users\mirio\research\vectors\CDR3_from_celiac_trim_3_4_with_labels_unique_sequences_Celiac_model_April_2020_VECTORS.csv'
Output_vector_file_path = r'C:\Users\mirio\research\vectors\CDR3_from_celiac_trim_3_4_with_labels_unique_sequences_Celiac_model_April_2020_VECTORS_VERIFY_10K_per_subject.csv'
Subject_field = 'FILENAME'

Input_data_file = pd.read_csv(Input_filtered_data_file_path)
Input_vectors_file = pd.read_csv(Input_vector_file_path)

by_subject = Input_data_file.groupby([Subject_field])
sub_num = 0
positive_subjects = 0

output_data_df = pd.DataFrame(columns = list(Input_data_file.columns))
output_vectors_df = pd.DataFrame(columns = list(Input_vectors_file.columns))


for subject, frame in by_subject:
    sub_num += 1
    print("------------------------")
    print(f"Analysing {subject!r} {sub_num!r}")
    dateTimeObj = datetime.now()
    timeObj = dateTimeObj.time()
    print(timeObj)
    print('Original number of sequences: ', len(frame), end="\n\n")
    # sample subject without replacement and save to list
    if len(frame) >= max_num_smaples_per_subject:
        print('High number of sequences, subject removed.\n')
    else:
        positive_subjects += 1
#        sampled_frame_indexes = np.random.choice(frame.index, max_num_smaples_per_subject, replace = False)
#        print(sampled_frame_indexes[:5])
        output_data_df = output_data_df.append(Input_data_file.iloc[frame.index])
        output_vectors_df = output_vectors_df.append(Input_vectors_file.iloc[frame.index])
        print('Subject added.')
        
        
    
print('----- Summary -----')
print(f'Analysed {sub_num!r} subjects')
print(f'Chose {positive_subjects!r} subjects with <= {max_num_smaples_per_subject} number of sequences')

output_data_df.to_csv(Output_filtered_data_file_path)
print('output data set saved to: ', Output_filtered_data_file_path)
output_vectors_df.to_csv(Output_vector_file_path, index = False)
print('output vectors set saved to: ', Output_vector_file_path)


