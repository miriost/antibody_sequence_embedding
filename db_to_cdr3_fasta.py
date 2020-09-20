#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import sys
import getopt


def main(argv):
	ifile = None
	ofile = None
	colname = None
	try:
		opts, args = getopt.getopt(argv, "hi:o:c:", ["input-file=", "output-file", "col-name"])
	except getopt.GetoptError:
		print('db_to_cdr3_fasta.py -i <input-file> -o <output-file> -c <colname>')
		sys.exit(2)
	reading_frame = None
	for opt, arg in opts:
		if opt == '-h':
			print('db_to_cdr3_fasta.py -i <inputfile> -o <outputfile> -c <colname>')
			sys.exit()
		elif opt in ("-i", "--input-file"):
			ifile = arg
		elif opt in ("-o", "--output-file"):
			ofile = arg
		elif opt in ("-c", "--col-name"):
			colname = arg

	if ifile is None or ofile is None or colname is None:
		print('db_to_cdr3_fasta.py -i <input-file> -o <output-file> -c <colname>')
		sys.exit(2)

	db = pd.read_csv(ifile)
	f = open(ofile, "w")
	for i in range(0,len(db)):
		f.write(">{}_CDR3\n{}\n".format(i, db.loc[i, colname]))
	f.close()

	print('output file {}'.format(ofile))


if __name__ == "__main__":
	main(sys.argv[1:])
