#!/bin/bash

trap "exit" INT

usage="USAGE: clustring_pipline.sh -v <vector_column> -l <labels> -f [folds] -c [cluster_sizes] -m [max_features] -w [work_dir]"
help="Build clusters and feature tables for train/test folds.
-v VECTOR_COLUMN - Mandatory, name of the column in the tsv file with the embedded vector.
-l LABELS - Mandatory, semicolon separated list of target labels for the clustering analysis and selection
-f FOLDS - Optional, space separated list of folds numbers. Deafult is 0..9.
-c CLUSTER_SIZES - Optional, space separated list of cluster sizes. Deafult is 100.
-m MAX_FEATURES - Optional, space separated list of max features to select per label. Default is 100.
-w WORK_DIR - Optional, the folds root directory where the folds are. Default is 50\"./\".
"
folds=$(seq 0 1 9)
cluster_sizes=100
significance_levels=70
max_features=50
vector_column=""
work_dir=./
labels=""

while getopts "hf:c:m:v:w:l:" opt; do
	case ${opt} in
		h ) echo "${usage}" ; echo "${help}"; exit 1
      			;;
    		f ) folds=${OPTARG}
      			;;
		c ) cluster_sizes=${OPTARG}
			      ;;
		m ) max_features=${OPTARG}
			      ;;
		v ) vector_column=${OPTARG}
			      ;;
		l ) labels=${OPTARG}
		        ;;
		w ) work_dir=${OPTARG}
			      ;;
		\? ) echo ${usage}; echo "cluster_pipline.sh -h for additional help"; exit 1
      			;;
	esac
done

# change to the working directory
cd ${work_dir}

# validate arguments
if [ -z "${vector_column}" ]; then
	echo "${usage}"
	echo "cluster_pipline.sh -h for additional help"
	exit -1
fi

if [ -z "${labels}" ]; then
	echo "${usage}"
	echo "cluster_pipline.sh -h for additional help"
	exit -1
fi


# loop folds
for fold in ${folds} ; do
	echo "Fold ${fold}"; echo ""
	fold_dir=FOLD${fold}
	
	# loop cluster size
	for cs in ${cluster_sizes}; do
		echo "Cluster size ${cs}"; echo ""
		cs_dir=${fold_dir}/cs_${cs}
		mkdir -p ${cs_dir}
		if [ -f ${cs_dir}/NN_cs_${cs}.tsv ] ; then
			echo "${cs_dir}/NN_cs_${cs}.tsv already exists, skipping KNN search."
		else
			# search K nearest neighbors
			echo "Starting KNN search and analysis..."
			eval python -u ~/antibody_sequence_embedding/executable_scripts/cluster_proximity_brute_force.py --data_file_path ${fold_dir}/*_TRAIN_*.tsv --perform_NN=True --perform_results_analysis=True --output_folder_path ${cs_dir} --output_description cs_${cs} --vector_column ${vector_column} --cluster_size ${cs} --cpus=12 --step=10000 --id repertoire.repertoire_name 2>&1 | tee -a ${cs_dir}/cs_${cs}_cluster_proximity_brute_force.log.txt
		fi
		if [ -f ${cs_dir}/cs_${cs}_analysis.csv ] ; then
			echo "${cs_dir}/cs_${cs}_analysis.csv already exists, skipping KNN analysis."
		else
			# analyze K nearest neighbors
			echo "Starting KNN analysis..."
			eval python -u ~/antibody_sequence_embedding/executable_scripts/cluster_proximity_brute_force.py --data_file_path ${fold_dir}/*_TRAIN_*.tsv --NN_file_path=${cs_dir}/NN_cs_${cs}.tsv --perform_NN=False --perform_results_analysis=True --output_folder_path ${cs_dir} --vector_column ${vector_column} --output_description cs_${cs} --cluster_size ${cs} --thread_memory 11474836480 --cpus=12 --step=10000 --id repertoire.repertoire_name 2>&1 | tee -a ${cs_dir}/cs_${cs}_cluster_proximity_brute_force.log.txt
		  mkdir -p ${cs_dir}/clustering_analysis
		  #python ~/antibody_sequence_embedding/executable_scripts/analyze_clustering.py --input_file ${cs_dir}/cs_${cs}_analysis.csv --labels "${labels}" --output_dir ${cs_dir}/clustering_analysis
		fi

		#loop max features
		for mf in ${max_features}; do 
			echo "Max features ${mf}"; echo ""
			output_dir=${cs_dir}/min_subj_5_max_features_${mf}
			mkdir -p ${output_dir}
			if [ -f ${output_dir}/feature_list.csv ] ; then
			echo "${output_dir}/feature_list.csv already exists, skipping building feature list." 
			else	
			# create feature list
			echo "Building feature list..."
			python -u ~/antibody_sequence_embedding/executable_scripts/build_cluster_proximty_feature_list.py --labels "${labels}" --data_file_path ${fold_dir}/*_TRAIN_*.tsv --analysis_file_path ${cs_dir}/cs_${cs}_analysis.csv --knn_file_path ${cs_dir}/NN_cs_${cs}.tsv --distances_file_path ${cs_dir}/Distances_cs_${cs}.csv --output_folder ${output_dir} --output_description feature_list --max_features ${mf} 2>&1 | tee ${output_dir}/build_cluster_proximity_feature_list.log.txt
			fi

			if [ -f ${output_dir}/train_feature_table.csv ] ; then
			echo "${output_dir}/train_feature_table.csv already exists, skipping building train feature table."
			else
			echo "Building train feature table..."
			# create train feature table
			python -u ~/antibody_sequence_embedding/executable_scripts/build_cluster_proximty_feature_table_with_labels.py --features_list ${output_dir}/feature_list.csv --data_file_path ${fold_dir}/*_TRAIN_*.tsv --vector_column ${vector_column} --output_folder ${output_dir} --output_description train --subject_col_name repertoire.repertoire_name --labels_col_name repertoire.disease_diagnosis --cpus=12 2>&1 | tee ${output_dir}/build_cluster_proximity_feature_table_with_labels.log.txt
			fi
			if [ -f ${output_dir}/test_feature_table.csv ] ; then
			echo "${output_dir}/test_feature_table.csv already exists, skipping building test feature table."	
			else
			echo "Building test feature table..."
			# create test feature table
			python -u ~/antibody_sequence_embedding/executable_scripts/build_cluster_proximty_feature_table_with_labels.py --labels "${labels}" --features_list ${output_dir}/feature_list.csv --data_file_path ${fold_dir}/*_TEST_*.tsv --vector_column ${vector_column} --output_folder ${output_dir} --output_description test --subject_col_name repertoire.repertoire_name --labels_col_name repertoire.disease_diagnosis --cpus=12 2>&1 | tee -a ${output_dir}/build_cluster_proximity_feature_table_with_labels.log.txt

			fi
		done # max features loop
	done # cluster size loop
done # fold loop
