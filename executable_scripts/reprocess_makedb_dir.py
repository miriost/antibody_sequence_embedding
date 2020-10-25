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

sys.path.insert(0, os.path.join(os.pardir, os.path.pardir))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_files', help='a space separated list of repertoire tsv files '
                                              'for db creation', nargs='+')
    parser.add_argument('--output_file', help='output file', type=str)
    parser.add_argument('--subject_field', help='output file', type=str, default='repertoire.repertoire_name')
    parser.add_argument('--trim', nargs=2, type=int, help='two integers to trim, first from beginning, 2nd from end')
    parser.add_argument('--min_seq_per_subject', type=int, help='min number of sequence per subject', default=10000)

    args = parser.parse_args()
    total_rows = 0

    df_list = []

    for file in args.input_files:
        if not (os.path.isfile(file)):
            print('cannot open {}.\nExiting...'.format(file))
            sys.exit(1)

        file_df = pd.read_csv(file, sep='\t')

        by_subject = file_df.groupby([args.subject_field])

        for subject, frame in by_subject:
        
            total_rows += len(frame)

            local_df = frame

            print('file {} sujbect {} total rows: {}'.format(file, subject, len(local_df)))

            # 1. REMOVE NON-FUNCTIONAL SEQUENCES
            local_df = local_df.loc[local_df.productive == True, :]
            print(' - After removing non functional sequences: {}'.format(len(local_df)))

            # 2. REMOVE ROWS WHERE JUNCTION LENGTH IS SHORTER THAN 12 OR DOESN'T DEVIDE BY 3
            local_df = local_df[local_df.junction_length > 12]
            local_df = local_df[local_df.junction_length % 3 == 0]
            print(' - After junction_length>12, junction_length % 3 == 0 : {}'.format(len(local_df)))

            # 3. LEAVE ONLY ROWS WHERE CONSCOUNT > 1
            local_df = local_df[local_df['consensus_count'] > 1]
            print(' - After consensus_count > 1: {} '.format(len(local_df)))

            # 4. REMOVE SEQUENCES WITH 'N' OR '-']
            local_df = local_df[local_df.junction.str.contains('N') == False]
            local_df = local_df[local_df.junction.str.contains('-') == False]
            print(' - After removing sequences with N or gaps: {}'.format(len(local_df)))

            # 5. trimming
            if args.trim:
                local_df = local_df[local_df.junction.str.len() - args.trim[0] - args.trim[1] >=3]
                print('trimming {} AAs from he beginning, {} AAs from to end'.format(args.trim[0], args.trim[1]))
                local_df['junction_aa_trim_' + str(args.trim[0]) + '_' + str(args.trim[1])] = \
                        local_df.junction_aa.str[args.trim[0]:-args.trim[1]]
                print(' - After trimming: {}'.format(len(local_df)))

            if len(local_df) >= args.min_seq_per_subject:
                df_list += [local_df]
            else:
                print('subject not added due to low number of sequences')

    if len(df_list) == 0:
        print("zero sequences where filtered")
        return

    big_df = pd.concat(df_list)

    big_df.to_csv(args.output_file, index=False, sep='\t')
    print('*********\nfile generated {} out of {} repertoires\nOriginal rows count: {} After filtering: {}'.format(
        args.output_file, len(big_df['repertoire.repertoire_name'].unique()), total_rows, len(big_df)))


if __name__ == "__main__":
    main()

