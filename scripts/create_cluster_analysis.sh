#!/bin/bash

trap "exit" INT

usage="USAGE: create_cluster_anlysis.sh -f [list of fold numbers] -c [list of cluster sizes]"
folds=$(seq 0 1 39)
cluster_sizes=$(seq 100 10 130)
while getopts "hf:c:s:m:o:" opt; do
	case ${opt} in
		h ) echo ${usage} ; exit 1
      			;;
    		f ) folds=${OPTARG}
      			;;
		c ) cluster_sizes=${OPTARG}
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
		output_dir=${fold_dir}/cs_${cs}
		mkdir -p ${output_dir}
		if [ -f ${output_dir}/feautre_list.csv ] ; then
			# already has a feature_list.csv file - skipping
			continue
		fi
		eval python -u ~/antibody_sequence_embedding/executable_scripts/cluster_proximity_brute_force.py --data_file_path ${fold_dir}/*_TRAIN_*.tsv --perform_NN=True --perform_results_analysis=True --output_folder_path ${output_dir} --vector_column celiac_light_chain_trim_1_1_prot2vec --output_description cs_${cs} --cluster_size ${cs} --thread_memory 11474836480 --cpus=12 --step=10000 --id repertoire.subject_id 2>&1 | tee -a ${output_dir}/cs_${cs}_cluster_proximity_brute_force.log.txt
	done
done 
