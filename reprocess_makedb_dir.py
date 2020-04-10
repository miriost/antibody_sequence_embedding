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
from Bio.Seq import Seq
from Bio.Alphabet import IUPAC


sys.path.insert(0, os.path.join(os.pardir, os.path.pardir))


def main(argv):
    
    parser=argparse.ArgumentParser(
            description='''reprocess_make_db.py script takes as input a directory containing makedb files and processes them to one file containing the fields: junction (AA), subject, condition\label ''',
            epilog="""All's well that ends well.""")
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', help='a directory containing makedb files')
    parser.add_argument('output_file', help='output file')
    parser.add_argument('--trim', nargs = 2, type = int, help = 'two integers to trim, first from beginning, 2nd from end')
    
    args = parser.parse_args()
    number_of_files = 0
    total_rows = 0
    filtered_rows = 0
    if not (os.path.isdir(args.input_dir)):
        print('Input directory {} error! Make sure the directory exists.\nExiting...'.format(args.input_dir))
        sys.exit(1)
    big_DF_ = []
    for filename in os.listdir(args.input_dir):
        if filename.endswith('_collapsed.tab'):
            
            print('~~~file #' + str(number_of_files) + ': ' + os.path.join(args.input_dir, filename))
            local_df = pd.read_table(os.path.join(args.input_dir, filename), usecols=['SEQUENCE_ID', 'FUNCTIONAL', 'JUNCTION_LENGTH', 'JUNCTION', 'DUPCOUNT', 'CONSCOUNT', 'V_CALL', 'D_CALL', 'J_CALL'], header = 0, index_col = None)
            total_rows += len(local_df)
            print('total rows: ' + str(len(local_df)))
            local_df['FILENAME'] = os.path.splitext(os.path.split(filename)[1])[0]
            ## DATA REPROCESS PART
            # 1. REMOVE NON-FUNCTIONAL SEQUENCES 
            
            local_df = local_df[local_df.FUNCTIONAL=='T']
    
            print(' - After removing non functional sequences: {}'.format(len(local_df)))
            # 2. REMOVE ROWS WHERE JUNCTION LENGTH IS SHORTER THAN 12 OR DOESN'T DEVIDE BY 3
            local_df = local_df[local_df.JUNCTION_LENGTH > 12]
            local_df = local_df[local_df.JUNCTION_LENGTH%3 == 0]
            print(' - After len>12, len%3==0 : {}'.format(len(local_df)))
            # 3. LEAVE ONLY ROWS WHERE CONSCOUNT > 1
            local_df = local_df[local_df['CONSCOUNT'] > 1]
            print(' - After CONSCOUNT > 1: {} '.format(len(local_df)))
            # 4. REMOVE SEQUENCES WITH 'N' OR '-'
            local_df['tmp'] = [junc if (('N' not in junc) and ('-' not in junc)) else 0 for junc in local_df.JUNCTION]
            local_df = local_df[local_df.tmp != 0]
            local_df = local_df.drop(['tmp'], axis = 1)
            print(' - After removing sequences with N or gaps: {}'.format(len(local_df)))
            filtered_rows += len(local_df)
    
            #  5. translate to AA and trim (if needed)
            
            local_df['JUNC_AA'] = [str(Seq(a).translate()) for a in local_df.JUNCTION]
            if args.trim:   
                print('trimmimg {} AAs from he beginning, {} AAs from te end'.format(args.trim[0], args.trim[1]))
                local_df['JUNC_AA'] = [a[args.trim[0]:-args.trim[1]] for a in local_df['JUNC_AA']]
            col_names = local_df.columns.values.tolist()
    
            if len(local_df) > 3000:
                big_DF_.append(local_df)
                number_of_files += 1
            else:
                print('file not added due to low number of sequences')
            
    comb_np_array = np.vstack(big_DF_)  
    big_frame = pd.DataFrame(comb_np_array)
    big_frame.columns = col_names
    #print(big_frame) 
    
    big_frame.to_csv(args.output_file, index = False)
    print('*********\nfile generated {} out of {} files\nOriginal rows count: {} After filtering: {}'.format(args.output_file, number_of_files, total_rows, filtered_rows))

        
if __name__ == "__main__":
   main(sys.argv[1:])       