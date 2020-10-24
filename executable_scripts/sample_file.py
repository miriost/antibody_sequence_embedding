# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 14:57:21 2020

@author: mirio
"""

import pandas as pd
import numpy as np
import argparse
import os


def main():

    parser = argparse.ArgumentParser(
        description='sample sequences by subjects from input file',
        epilog="All's well that ends well.")
    parser.add_argument('--min_samples', type=int, help='min number of sampled sequences per subject. Default is 10k.',
                        default=10000)
    parser.add_argument('--input_data_file', type=str, help='input filtered data file path')
    parser.add_argument('--seed', type=str, help='seed for the random sampling', default=0)
    parser.add_argument('--output_data_file', help='output filtered data file path', type=str)
    parser.add_argument('--subject_field', help='name of the column which identifies the subjects '
                                                '(default=repertoire.subject_id)', default='repertoire.subject_id')
    parser.add_argument('--exclude_dup_column', help='Filter subject level duplication by this '
                                                     'column (default=junction_aa)', default='junction_aa')

    args = parser.parse_args()

    input_data_file = pd.read_csv(args.input_data_file, sep='\t')
    # sort file before sampling to get consisted sampling in different runs (if input include same sequences
    # but not necessarily sorted)
    input_data_file.sort_index(inplace=True)

    by_subject = input_data_file.groupby([args.subject_field])
    sub_num = 0
    positive_subjects = 0

    np.random.seed(args.seed)

    df_list = []
    for subject, frame in by_subject:
        sub_num += 1
        print("------------------------")
        print(f"Analysing {subject!r} {sub_num!r}")
        print('Original number of sequences: ', len(frame), end="\n\n")
        # sample subject without replacement and save to list
        tmp_df = frame.drop_duplicates(subset=[args.exclude_dup_column])
        if len(tmp_df) < args.min_samples:
            print('Lower number of sequences {}, subject removed.\n'.format(len(tmp_df)))
            continue 
        positive_subjects += 1
        sampled_frame_indexes = np.random.choice(tmp_df.index, args.min_samples, replace=False)
        df_list += [input_data_file.iloc[sampled_frame_indexes]]
        print('Subject added.')

    if len(df_list) == 0:
        print("no sujbect passed filter")
        return

    output_data_df = pd.concat(df_list)

    print('----- Summary -----')
    print(f'Analysed {sub_num!r} subjects')
    print(f'Chose {positive_subjects!r} subjects with >= {args.min_samples} number of sequences')

    if args.output_data_file is None:
        file_name = os.path.basename(args.input_data_file).split(".tsv")[0]
        dir_name = os.path.dirname(args.input_data_file)
        output_file_name = os.path.join(dir_name, 
                file_name + '_sampled_n' + str(args.min_samples) + '_seed' + str(args.seed) + '.tsv')
    else:
        output_file_name = args.output_data_file

    output_data_df.to_csv(output_file_name, sep='\t', index=False)
    print('output data set saved to: ', output_file_name)


if __name__ == "__main__":
    main()

