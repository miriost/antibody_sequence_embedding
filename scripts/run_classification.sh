#!/bin/bash

trap "exit" INT


# loop folds
for fold in $(seq 0 1 39) ; do
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
				# runt the classification
				python ~/antibody_sequence_embedding/executable_scripts/classify_no_splitting.py --train_file ${output_dir}/train_feature_table.csv  --test_file ${output_dir}/test_feature_table.csv --col_names="fold,cluster_size,segnificant" --col_values="${fold},${cs},${sig}" --output_file ${output_dir}/classification_res.csv -M all
			done # min subjects loop
		done # segnificant level loop
	done # cluster size loop
done # fold loop
