#!/bin/bash

trap "exit" INT

lower=100
upper=130
while getopts "hl:u:s" opt; do
	case ${opt} in
    		h ) echo "USAGE: create_cluster_analysis.sh -l [lower fold number] -u [upper fold number]"
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
		output_dir=${fold_dir}/cs_${cs}
		mkdir -p ${output_dir}
		if [ -f ${output_dir}/feautre_list.csv ] ; then
			# already has a feature_list.csv file - skipping
			continue
		fi
		python -u ~/antibody_sequence_embedding/executable_scripts/cluster_proximity_brute_force.py -d ${fold_dir}/FILTERED_DATA_TRAIN.tab -v ${fold_dir}/VECTORS_TRAIN.csv --perform_NN=True --perform_results_analysis=True -of ${output_dir} -od cs_${cs} -cs ${cs} -tm 11474836480 --cpus=12 --step=10000 -id FILENAME 2>&1 | tee -a ${output_dir}/cs_${cs}_cluster_proximity_brute_force.log.txt
	done
done 
