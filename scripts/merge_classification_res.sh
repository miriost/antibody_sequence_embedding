#!/bin/bash

trap "exit" INT

usage="USAGE: merge_classification_res.sh -f [list of fold numbers] -c [list of cluster sizes] -s [list of segnificant levels] -m [list of min subjects] -o [classification res file name]"
folds=$(seq 0 1 39)
cluster_sizes=$(seq 100 10 130)
segnificant_levels=$(seq 60 2 74)
min_subjects=$(seq 8 1 14)
classification_file=classification_res.csv
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
		o ) classification_file=${OPTARG}
			;;
		\? ) echo ${usage}; exit 1
      			;;
	esac
done

# reset file if exists
echo "" > all_${classification_file}
# loop all folds
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
				if ! [ -f ${output_dir}/${classification_file} ]; then
					continue
				fi
				echo "merging ${output_dir}/${classification_file}"
				tail -n+2 ${output_dir}/${classification_file} >> all_${classification_file}	
			done		
		done
	done
done 
