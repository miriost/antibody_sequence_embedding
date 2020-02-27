#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 13:34:35 2018

@author: miri-o
"""
import pandas as pd
original_file = pd.read_csv('/media/miri-o/Documents/filtered_data_sets/Celiac_for_V_family_analysis_3M_seqs_Celiac_n_3_trimming_2_1_labeled_FILTERED_DATA.csv', sep = ',')
print(original_file.groupby('V_FAMILY').size())
