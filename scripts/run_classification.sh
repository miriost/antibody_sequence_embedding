#!/bin/bash

trap "exit" INT

usage="USAGE: create_feature_list.sh -f [list of fold numbers] -c [list of cluster sizes] -s [list of segnificant levels] -m [list of min subjects] -o [output file name]"
folds=$(seq 0 1 39)
cluster_sizes=$(seq 100 10 130)
segnificant_levels=$(seq 60 2 74)
min_subjects=$(seq 8 1 14)
output_file=classification_res.csv
while getopts "hf:c:s:m:o:" opt; do
	case ${opt} in
		h ) echo ${usage} ; exit 1
      			;;
    		f ) folds=${OPTARG}
      			;;
		c ) cluster_sizes=${OPTARG}
			;;
    		s ) segnificant_levels=${OPTARG}
		     	;;
		m ) min_subjects=${OPTARG}
			;;
		o ) output_file=${OPTARG}
			;;
		\? ) echo ${usage}; exit 1
      			;;
	esac
done

# loop folds
for fold in ${folds} ; do
	fold_dir=FOLD${fold}
	# loop cluster size
	for cs in ${cluster_sizes}; do
		cs_dir=${fold_dir}/cs_${cs}
		# loop segnificant level
		for seg_level in ${segnificant_levels} ; do
			sig=$(echo "scale=2;${seg_level}/100" | bc)
			# loop min subjects
			for min_subj in ${min_subjects}; do 
				output_dir=${cs_dir}/seg_level_${seg_level}_min_subj_${min_subj}
				# runt the classification
				python -u ~/antibody_sequence_embedding/executable_scripts/classify_no_splitting.py --train_file ${output_dir}/train_feature_table.csv  --test_file ${output_dir}/test_feature_table.csv --col_names="fold,cluster_size,segnificant" --col_values="${fold},${cs},${sig}" --output_file ${output_dir}/${output_file} -M all 2>&1 | tee ${output_dir}/classifiy_no_splitting.log.txt
			done # min subjects loop
		done # segnificant level loop
	done # cluster size loop
done # fold loop
