import pandas as pd
import numpy as np
import sys
import argparse
import os


def main():
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--input_file', help='')
	parser.add_argument('--output_file', help='')
	parser.add_argument('--label_col', help='')
	parser.add_argument('--id_col', help='')

	args = parser.parse_args()
	execute(args)


def execute(args):

	df = pd.read_csv(args.input_file, sep='\t')
	subjects = pd.DataFrame()
	subjects['id'] = df[args.id_col].unique().tolist()
	subjects['label'] = [df.loc[df[args.id_col] == x, args.label_col][0] for x in subjects['id']]
	subjects['label'] = np.random.permutation(subjects['label'].values)

	for idx, row in subjects.iterrows():
		df.loc[df[args.id_col] == row['id'], args.label_col] = row['label']

	df.to_csv(args.output_file, '\t')


if __name__ == '__main__':
	main()
