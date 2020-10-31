#!/bin/bash

trap "exit" INT

usage="USAGE: clustring_pipline.sh -v <vector_column> -f [folds] -c [cluster_sizes] -s [significance] -m [min_subjects] -w [work_dir]"
help="Build clusters and feature tables for train/test folds.
-v VECTOR_COLUMN - Mandatory, name of the column in the tsv file with the embedded vector.
-f FOLDS - Optional, space separated list of folds numbers. Deafult is 0..9.
-c CLUSTER_SIZES - Optional, space separated list of cluster sizes. Deafult is 80 100 120.
-s SEGNIFICANCE - Optional, space separated list of min segnificance precentages for the feature (cluster) selection. Default is 60 63 66.
-m MIN_SUBJECTS - Optional, space separated list of min subjects for the feature (cluster) selection. Default is 10.
-w WORK_DIR - Optional, the folds root directory where the folds are. Default is \"./\".
"
folds=$(seq 0 1 9)
cluster_sizes=$(seq 80 20 120)
significance_levels=$(seq 60 3 66)
min_subjects=10
vector_column=""
work_dir=./

while getopts "hf:c:s:m::v:w:" opt; do
	case ${opt} in
		h ) echo "${usage}" ; echo "${help}"; exit 1
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
		w) work_dir=${OPTARG}
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

# loop folds
for fold in ${folds} ; do
	echo "Fold ${fold}"; echo ""
	fold_dir=FOLD${fold}
	
	# loop cluster size
	for cs in ${cluster_sizes}; do
		echo "Cluster size ${cs}"; echo ""
		cs_dir=${fold_dir}/cs_${cs}
		mkdir -p ${cs_dir}
		if [ -f ${cs_dir}/NN_cs_${cs}.csv ] ; then
			echo "${cs_dir}/NN_cs_${cs}.csv already exists, skipping KNN search."
		else
			# search K nearest neighbors
			echo "Starting KNN search and analysis..."
			eval python -u ~/antibody_sequence_embedding/executable_scripts/cluster_proximity_brute_force.py --data_file_path ${fold_dir}/*_TRAIN_*.tsv --perform_NN=True --perform_results_analysis=True --output_folder_path ${cs_dir} --vector_column ${vector_column} --output_description cs_${cs} --cluster_size ${cs} --cpus=12 --step=10000 --id repertoire.repertoire_name 2>&1 | tee -a ${cs_dir}/cs_${cs}_cluster_proximity_brute_force.log.txt
		fi
		if [ -f ${cs_dir}/cs_${cs}_analysis.csv ] ; then
			echo "${cs_dir}/cs_${cd}_analysis.csv already exists, skipping KNN analysis."
		else
			# analyze K nearest neighbors
			echo "Starting KNN analysis..."
			eval python -u ~/antibody_sequence_embedding/executable_scripts/cluster_proximity_brute_force.py --data_file_path ${fold_dir}/*_TRAIN_*.tsv --NN_file_path=${output_dir}/NN_cs_${cs}.csv --perform_NN=False --perform_results_analysis=True --output_folder_path ${cs_dir} --vector_column ${vector_column} --output_description cs_${cs} --cluster_size ${cs} --thread_memory 11474836480 --cpus=12 --step=10000 --id repertoire.repertoire_name 2>&1 | tee -a ${cs_dir}/cs_${cs}_cluster_proximity_brute_force.log.txt
		fi

		# loop significance level
		for sig_level in ${significance_levels} ; do
			sig=$(echo "scale=2;${sig_level}/100" | bc)
			echo "Significance ${sig}"; echo ""
			
			# loop min subjects
			for min_subj in ${min_subjects}; do 
				echo "Min subjects ${min_subj}"; echo ""
				output_dir=${cs_dir}/sig_level_${sig_level}_min_subj_${min_subj}
				mkdir -p ${output_dir}
				if [ -f ${output_dir}/feature_list.csv ] ; then
					echo "${output_dir}/feature_list.csv already exists, skipping building feature list." 
				else	
					# create feature list
					echo "Building feature list..."
					python -u ~/antibody_sequence_embedding/executable_scripts/build_cluster_proximty_feature_list.py --data_file_path ${fold_dir}/*_TRAIN_*.tsv --analysis_file_path ${cs_dir}/cs_${cs}_analysis.csv --distances_file_path ${cs_dir}/Distances_cs_${cs}.csv --vector_column ${vector_column} --output_folder ${output_dir} --output_description feature_list --label_freq_col Healthy --significance ${sig} --min_subjects ${min_subj} 2>&1 | tee ${output_dir}/build_cluster_proximity_feature_list.log.txt
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
					python -u ~/antibody_sequence_embedding/executable_scripts/build_cluster_proximty_feature_table_with_labels.py --features_list ${output_dir}/feature_list.csv --data_file_path ${fold_dir}/*_TEST_*.tsv --vector_column ${vector_column} --output_folder ${output_dir} --output_description test --subject_col_name repertoire.repertoire_name --labels_col_name repertoire.disease_diagnosis --cpus=12 2>&1 | tee -a ${output_dir}/build_cluster_proximity_feature_table_with_labels.log.txt

				fi
			done # min subjects loop
		done # significance level loop
	done # cluster size loop
done # fold loop
