#!/bin/bash

cd /media/miri-o/Documents/miris_tools/executable_scripts/
filename=/media/miri-o/Documents/vectors/FLU_data_012119_celiac_model_Jan19_2019_VECTORS.csv

for dim in 2 5 10 15 20 30 50 75 90 100; 
do
	echo ~~~~~ $dim
	python reduce_dimensions.py -i $filename -o /media/miri-o/Documents/vectors/FLU_data_012119_$dim"D_PCA.csv" -n $dim -m PCA
done
