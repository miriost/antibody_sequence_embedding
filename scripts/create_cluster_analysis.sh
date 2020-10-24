#!/bin/bash

trap "exit" INT

usage="USAGE: create_cluster_anlysis.sh -v <vector column name> -f [list of fold numbers] -c [list of cluster sizes]"
folds=$(seq 0 1 39)
cluster_sizes=$(seq 100 10 130)
vector_column=false
while getopts "hf:c:v:" opt; do
	case ${opt} in
		h ) echo ${usage} ; exit 1
      			;;
    		f ) folds=${OPTARG}
      			;;
		c ) cluster_sizes=${OPTARG}
			;;
		v ) vector_column=${OPTARG}
			;;
		\? ) echo ${usage}; exit 1
      			;;
	esac
done

if [ vector_column == false ]; then
	echo ${usage}
	exit -1
fi	

# loop folds
for fold in ${folds} ; do
	fold_dir=FOLD${fold}
	# loop cluster size
	for cs in ${cluster_sizes}; do
		output_dir=${fold_dir}/cs_${cs}
		mkdir -p ${output_dir}
		if [ -f ${output_dir}/feautre_list.csv ] ; then
			# already has a feature_list.csv file - skipping
			continue
		fi
		if [ -f ${output_dir}/NN_cs_${cs}.csv ] ; then
			eval python -u ~/antibody_sequence_embedding/executable_scripts/cluster_proximity_brute_force.py --data_file_path ${fold_dir}/*_TRAIN_*.tsv --NN_file_path=${output_dir}/NN_cs_${cs}.csv  --perform_NN=False --perform_results_analysis=True --output_folder_path ${output_dir} --vector_column ${vector_column} --output_description cs_${cs} --cluster_size ${cs} --thread_memory 11474836480 --cpus=12 --step=10000 --id repertoire.subject_id 2>&1 | tee -a ${output_dir}/cs_${cs}_cluster_proximity_brute_force.log.txt
	
		else
			eval python -u ~/antibody_sequence_embedding/executable_scripts/cluster_proximity_brute_force.py --data_file_path ${fold_dir}/*_TRAIN_*.tsv --perform_NN=True --perform_results_analysis=True --output_folder_path ${output_dir} --vector_column ${vector_column} --output_description cs_${cs} --cluster_size ${cs} --thread_memory 11474836480 --cpus=12 --step=10000 --id repertoire.subject_id 2>&1 | tee -a ${output_dir}/cs_${cs}_cluster_proximity_brute_force.log.txt
		fi
	done
done 
