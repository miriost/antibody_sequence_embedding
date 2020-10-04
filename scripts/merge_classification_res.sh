#!/bin/bash

trap "exit" INT

# loop folds
for fold in 0 1 2 3 ; do
	fold_dir=FOLD${fold}
	# loop cluster size
	for cs in seq(100 10 130); do
		cs_dir=${fold_dir}/cs_${cs}
		# loop segnificant level
		for seg_level in seq(60 2 74) ; do
			sig=$(echo "scale=2;${seg_level}/100" | bc)
			# loop min subjects
			for min_subj in seq(8 1 14); do 
				output_dir=${cs_dir}/seg_level_${seg_level}_min_subj_${min_subj}
				if ! [ -f ${output_dir}/classification_res.csv ]; then
					continue
				fi
				echo "merging ${output_dir}/classification_res.csv"
				if ! [ -f all_classification_res.csv ]; then
					cp ${output_dir}/classification_res.csv all_classification_res.csv
		       		else
					tail -n+2 ${output_dir}/classification_res.csv >> all_classification_res.csv 	
				fi
			done		
		done
	done
done 
