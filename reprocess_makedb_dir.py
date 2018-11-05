#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reprocess makedb directory into one file cotaining junction (AA), subject, condition\label
Created on Mon Nov  5 11:46:26 2018

@author: miri-o
"""

import sys, argparse
import os
import pandas as pd
import numpy as np


sys.path.insert(0, os.path.join(os.pardir, os.path.pardir))


def main(argv):
    
    parser=argparse.ArgumentParser(
            description='''reprocess_make_db.py script takes as input a directory containing makedb files and processes them to one file containing the fields: junction (AA), subject, condition\label ''',
            epilog="""All's well that ends well.""")
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', help='a directory containing makedb files')
    parser.add_argument('output_file', help='output file')
    
    args = parser.parse_args()
    number_of_files = 0
    total_rows = 0
    if not (os.path.isdir(args.input_dir)):
        print('Input directory {} error! Make sure the directory exists.\nExiting...'.format(args.input_dir))
        sys.exit(1)
    big_DF_ = []
    for filename in os.listdir(args.input_dir):
        if filename.endswith('pass_collapsed.tab'):
            number_of_files += 1
            print('~~~file #' + str(number_of_files) + ': ' + os.path.join(args.input_dir, filename))
            local_df = pd.read_table(os.path.join(args.input_dir, filename), usecols=['SEQUENCE_ID', 'FUNCTIONAL', 'JUNCTION_LENGTH', 'JUNCTION', 'DUPCOUNT'], header = 0, index_col = None)
            total_rows += len(local_df)
            print('total rows: ' + str(len(local_df)))
            local_df['FILENAME'] = os.path.splitext(os.path.split(filename)[1])[0]

            big_DF_.append(local_df)
            
    comb_np_array = np.vstack(big_DF_)  
    big_frame = pd.DataFrame(comb_np_array)
    big_frame.columns = ['SEQUENCE_ID', 'FUNCTIONAL', 'JUNCTION_LENGTH', 'JUNCTION', 'DUPCOUNT', 'FILENAME']
    #print(big_frame) 
    ## DATA REPROCESS PART
    # 1. REMOVE NON-FUNCTIONAL SEQUENCES 
    print(' - Begining data reprocess and filterting, number of original rows: ' +str(total_rows))
    big_frame = big_frame[big_frame.FUNCTIONAL=='T']
    current_len = len(big_frame)
    print('Removed non functional {} sequences'.format(total_rows-current_len))
    # 2. REMOVE ROWS WHERE JUNCTION LENGTH IS SHORTER THAN 12 OR DOESN'T DEVIDE BY 3
    big_frame = big_frame[big_frame.JUNCTION_LENGTH > 12]
    big_frame = big_frame[big_frame.JUNCTION_LENGTH%3 == 0]
    print(' - Removed {} short junctions '.format(current_len-len(big_frame)))
    current_len = len(big_frame)
    
    
    big_frame.to_csv(args.output_file, index = False)
    print('file generated {} out of {} files\nOriginal rows count: {} After filtering: {}'.format(args.output_file, number_of_files, total_rows, len(big_frame)))

        
if __name__ == "__main__":
   main(sys.argv[1:])       