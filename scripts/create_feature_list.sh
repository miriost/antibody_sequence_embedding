#!/bin/bash

trap "exit" INT

lower=100
upper=130
while getopts "hl:u:s" opt; do
	case ${opt} in
		h ) echo "USAGE: create_feature_list.sh -l [lower fold number] -u [upper fold number]"
      			;;
    		l ) lower=${OPTARG}
      			;;
    		u ) upper=${OPTARG}
		     	;;
		\? ) echo $USAGE; exit 1
      		;;
	esac
done

# loop folds
for fold in $(seq ${lower} 1 ${upper}) ; do
	fold_dir=FOLD${fold}
	# loop cluster size
	for cs in $(seq 100 10 130); do
		cs_dir=${fold_dir}/cs_${cs}
		# loop segnificant level
		for seg_level in $(seq 60 2 74) ; do
			sig=$(echo "scale=2;${seg_level}/100" | bc)
			# loop min subjects
			for min_subj in $(seq 8 1 14); do 
				output_dir=${cs_dir}/seg_level_${seg_level}_min_subj_${min_subj}
				mkdir -p ${output_dir}
				if ! [ -f ${output_dir}/feature_list.csv ] ; then 
					# create feature list
					python -u ~/antibody_sequence_embedding/executable_scripts/build_cluster_proximty_feature_list.py -a ${cs_dir}/cs_${cs}_analysis.csv -ds ${cs_dir}/Distances_cs_${cs}.csv -v ${fold_dir}/VECTORS_TRAIN.csv -of ${output_dir} -od feature_list -l HC -S ${sig} -m ${min_subj} 2>&1 | tee ${output_dir}/build_cluster_proximty_feature_list.log.txt
				fi
				if ! [ -f ${output_dir}/train_feature_table.csv ] ; then
					# create train feature table
					python -u ~/antibody_sequence_embedding/executable_scripts/build_cluster_proximty_feature_table_with_labels.py -f ${output_dir}/feature_list.csv -d ${fold_dir}/FILTERED_DATA_TRAIN.tab -v ${fold_dir}/VECTORS_TRAIN.csv -of ${output_dir} -od train -s FILENAME -l labels --cpus=12 2>&1 | tee ${output_dir}/build_cluster_proximity_feature_table_with_labels.log.txt
				fi
				if ! [ -f ${output_dir}/test_feature_table.csv ] ; then
					# create test feature table
					python -u ~/antibody_sequence_embedding/executable_scripts/build_cluster_proximty_feature_table_with_labels.py -f ${output_dir}/feature_list.csv -d ${fold_dir}/FILTERED_DATA_TEST.tab -v ${fold_dir}/VECTORS_TEST.csv -of ${output_dir} -od test -s FILENAME -l labels --cpus=12 2>&1 | tee -a ${output_dir}/build_cluster_proximity_feature_table_with_labels.log.txt
				fi
			done # min subjects loop
		done # segnificant level loop
	done # cluster size loop
done # fold loop
