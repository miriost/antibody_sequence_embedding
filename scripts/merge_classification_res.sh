#!/bin/bash

trap "exit" INT

usage="USAGE: merge_classification_res.sh -f [list of fold numbers] -c [list of cluster sizes] -s [list of significance levels] -m [list of min subjects] -o [classification res file name]"
folds=$(seq 0 1 39)
cluster_sizes=$(seq 100 10 130)
significance_levels=$(seq 60 2 74)
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
    		s ) significance_levels=${OPTARG}
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
rm -f all_${classification_file}
# loop all folds
for fold in ${folds} ; do
	fold_dir=FOLD${fold}
	# loop cluster size
	for cs in ${cluster_sizes}; do
		cs_dir=${fold_dir}/cs_${cs}
		# loop significance level
		for sig_level in ${significance_levels} ; do
			# loop min subjects
			for min_subj in ${min_subjects}; do 
				output_dir=${cs_dir}/sig_level_${sig_level}_min_subj_${min_subj}
				if ! [ -f ${output_dir}/${classification_file} ]; then
					continue
				fi
				if ! [ -f all_${classification_file} ] ; then
					head -n1 ${output_dir}/${classification_file} > all_${classification_file}	 
				fi	
				echo "merging ${output_dir}/${classification_file}"
				tail -n+2 ${output_dir}/${classification_file} >> all_${classification_file}	
			done		
		done
	done
done 
