#!/bin/bash

models=/media/miri-o/Documents/Immune2vec/trained_models/*.model
filename=/media/miri-o/Documents/CDR3_properties_HCV.csv

for modelname in $models; do
	python /media/miri-o/Documents/python_scripts/genrate_vectors.py $filename $modelname
done
