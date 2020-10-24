#!/bin/bash

trap "exit" INT

usage="USAGE: create_feature_list.sh -v <vector_column> -f [list of fold numbers] -c [list of cluster sizes] -s [list of significance levels] -m [list of min subjects]"
folds=$(seq 0 1 39)
cluster_sizes=$(seq 100 10 130)
significance_levels=$(seq 60 2 74)
min_subjects=$(seq 8 1 14)
vector_column=false
while getopts "hf:c:s:m:o:v:" opt; do
	case ${opt} in
		h ) echo ${usage} ; exit 1
      			;;
    		f ) folds=${OPTARG}
      			;;
		c ) cluster_sizes=${OPTARG}
			;;
    		s ) significance_levels=${OPTARG}
		     	;;
		m ) min_subjects=${OPTARG}
			;;
		v) vector_column=${OPTARG}
			;;
		\? ) echo ${usage}; exit 1
      			;;
	esac
done


if [ ${vector_column} == false ]; then
	echo ${usage}
	exit 1
fi

# loop folds
for fold in ${folds} ; do
	fold_dir=FOLD${fold}
	# loop cluster size
	for cs in ${cluster_sizes}; do
		cs_dir=${fold_dir}/cs_${cs}
		# loop significance level
		for sig_level in ${significance_levels} ; do
			sig=$(echo "scale=2;${sig_level}/100" | bc)
			# loop min subjects
			for min_subj in ${min_subjects}; do 
				output_dir=${cs_dir}/sig_level_${sig_level}_min_subj_${min_subj}
				mkdir -p ${output_dir}
				if ! [ -f ${output_dir}/feature_list.csv ] ; then 
					# create feature list
					python -u ~/antibody_sequence_embedding/executable_scripts/build_cluster_proximty_feature_list.py --data_file_path ${fold_dir}/*_TRAIN_*.tsv --analysis_file_path ${cs_dir}/cs_${cs}_analysis.csv --distances_file_path ${cs_dir}/Distances_cs_${cs}.csv --vector_column ${vector_column} --output_folder ${output_dir} --output_description feature_list --label_freq_col Healthy --significance ${sig} --min_subjects ${min_subj} 2>&1 | tee ${output_dir}/build_cluster_proximty_feature_list.log.txt
				fi
				if ! [ -f ${output_dir}/train_feature_table.csv ] ; then
					# create train feature table
					python -u ~/antibody_sequence_embedding/executable_scripts/build_cluster_proximty_feature_table_with_labels.py --features_list ${output_dir}/feature_list.csv --data_file_path ${fold_dir}/*_TRAIN_*.tsv --vector_column ${vector_column} --output_folder ${output_dir} --output_description train --subject_col_name repertoire.subject_id --labels_col_name repertoire.disease_diagnosis --cpus=12 2>&1 | tee ${output_dir}/build_cluster_proximity_feature_table_with_labels.log.txt
				fi
				if ! [ -f ${output_dir}/test_feature_table.csv ] ; then
					# create test feature table
					python -u ~/antibody_sequence_embedding/executable_scripts/build_cluster_proximty_feature_table_with_labels.py --features_list ${output_dir}/feature_list.csv --data_file_path ${fold_dir}/*_TEST_*.tsv --vector_column ${vector_column} --output_folder ${output_dir} --output_description test --subject_col_name repertoire.subject_id --labels_col_name repertoire.disease_diagnosis --cpus=12 2>&1 | tee -a ${output_dir}/build_cluster_proximity_feature_table_with_labels.log.txt

				fi
			done # min subjects loop
		done # significance level loop
	done # cluster size loop
done # fold loop
