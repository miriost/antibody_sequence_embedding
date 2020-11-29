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
from Levenshtein import distance

sys.path.insert(0, os.path.join(os.pardir, os.path.pardir))


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', "True"):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', "False"):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='input tsv file')
    parser.add_argument('output_file', help='output file', type=str)
    parser.add_argument('--naive', type=str2bool, help='sequences are of naive cells, default is False.', default=False)
    parser.add_argument('--min_seq_per_subject', type=int, help='min number of sequence per subject, default is 2000',
                        default=2000)

    args = parser.parse_args()
    total_rows = 0
    input_file = args.input_file
    id_column = 'subject.subject_id'

    df_list = []

    if not (os.path.isfile(input_file)):
        print('cannot open {}.\nExiting...'.format(input_file))
        sys.exit(1)

    file_df = pd.read_csv(input_file, sep='\t')
    file_df['cdr3_length'] = file_df['cdr3'].str.len()
    file_df['cdr3_aa_length'] = file_df['cdr3_aa'].str.len()

    by_subject = file_df.groupby([id_column])

    for subject, frame in by_subject:

        total_rows += len(frame)

        local_df = frame

        print('sujbect {} total rows: {}'.format(subject, len(local_df)))

        # 1. REMOVE NON-FUNCTIONAL SEQUENCES
        local_df = local_df.loc[local_df.productive == True, :]
        print(' - After removing non functional sequences: {}'.format(len(local_df)))

        # 2. REMOVE ROWS WHERE CDR3 LENGTH DOESN'T DIVIDE BY 3
        local_df = local_df[local_df.cdr3_length % 3 == 0]
        print(' - After cdr3_length % 3 == 0 : {}'.format(len(local_df)))

        # 3. LEAVE ONLY ROWS WHERE CONSCOUNT > 1
        local_df = local_df[local_df['consensus_count'] > 1]
        print(' - After consensus_count > 1: {} '.format(len(local_df)))

        # 4. REMOVE SEQUENCES WITH EDIT DISTANCE > 3 FROM GERMLINE
        if args.naive:
            dist_from_germline = local_df.apply(lambda x: distance(x['v_germline_alignment'],
                                                                   x['v_sequence_alignment']), axis=1)
            local_df = local_df[dist_from_germline <= 3]
            print(' - After dist_from_germline <= 3: {} '.format(len(local_df)))

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

