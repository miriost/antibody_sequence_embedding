#!/bin/bash

cd /media/miri-o/Documents/miris_tools/executable_scripts/
filename=/media/miri-o/Documents/vectors/CDR3_from_Celiac_full-2018-05-10_Celiac_n_3_trimming_2_1_VECTORS.csv

for dim in {2..100}; 
do
	echo ~~~~~ $dim
	python reduce_dimensions.py -i $filename -o /media/miri-o/Documents/vectors/CDR3_from_Celiac_full-2018-05-10_Celiac_n_3_trimming_2_1_VECTORS_$dim"D_PCA.csv" -n $dim -m PCA
done
