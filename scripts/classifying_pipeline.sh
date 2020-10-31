#!/bin/bash

trap "exit" INT

usage="USAGE: run_classification.sh -f [folds] -c [cluster_sizes] -s [significance] -m [min_subjects] -o [output_file] -r [replace] -M [model] -g [grid_serach] -w [work_dir]"
help="
-f FOLDS - Optional, space separated list of folds numbers. Deafult is 0..9.
-c CLUSTER_SIZES - Optional, space separated list of cluster sizes. Deafult is 80 100 120.
-s SEGNIFICANCE - Optional, space separated list of min segnificance precentages for the feature (cluster) selection. Default is 60 63 66.
-m MIN_SUBJECTS - Optional, space separated list of min subjects for the feature (cluster) selection. Default is 10.
-M MODEL - Optional, which model to use for the classification. Default is all.
-o OUTPUT_FILE - Optional, classification results output file name. Default is classification_res.csv
-r REPLACE - Optional, override existing results file. Default is false.
-g GRID_SEARCH - Use grid search over repeated cross validation folds for optimizing the classsifier hyper parameters. Default is false.
-w WORK_DIR - Optional, the folds root directory where the folds are. Default is \"./\".
"

folds=$(seq 0 1 9)
cluster_sizes=$(seq 80 20 120)
significance_levels=$(seq 60 3 66)
min_subjects=$(seq 5)
output_file=classification_res.csv
replace="false"
models=all
optimize="false"
work_dir=./

while getopts "hf:c:s:m:o:r:M:g:w:" opt; do
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
		o ) output_file=${OPTARG}
			;;
		r ) replace=${OPTARG}
			;;
		M ) models=${OPTARG}
			;;
		g ) optimize=${OPTARG}
			;;
		w ) work_dir=${OPTARG}
			;;
		\? ) echo ${usage}; echo "classifiying_pipeline.sh -h for additional help."; exit 1
      			;;
	esac
done

# change to working dir
cd ${work_dir}

# loop folds
for fold in ${folds} ; do
	echo "Fold ${fold}"
	fold_dir=FOLD${fold}
	# loop cluster size
	for cs in ${cluster_sizes}; do
		echo "Cluster size ${cs}"
		cs_dir=${fold_dir}/cs_${cs}
		# loop significance level
		for sig_level in ${significance_levels} ; do
			sig=$(echo "scale=2;${sig_level}/100" | bc)
			echo "Segnificance ${sig}"
			# loop min subjects
			for min_subj in ${min_subjects}; do 
				echo "Min subject ${min_subj}"
				output_dir=${cs_dir}/sig_level_${sig_level}_min_subj_${min_subj}
				if [ -f ${output_dir}/${output_file} ] && [[ "${replace}" == "false" ]] ; then
					echo "file ${output_dir}/${output_file} already exists, skipping classification."
					continue
				fi
				if ! [ -d ${output_dir}  ] ; then
					echo "directory ${output_dir} does not exits, skipping."
					continue
				fi
				# runt the classification
				echo "Classifying..."
				python -u ~/antibody_sequence_embedding/executable_scripts/classify_no_splitting.py --train_file ${output_dir}/train_feature_table.csv  --test_file ${output_dir}/test_feature_table.csv --col_names="min_subj,fold,cluster_size,significance" --col_values="${min_subj},${fold},${cs},${sig}" --output_file ${output_dir}/${output_file} -M ${models} --grid_search=${optimize} 2>&1 | tee ${output_dir}/classifiy_no_splitting.log.txt
			done # min subjects loop
		done # significance level loop
	done # cluster size loop
done # fold loop
#!/bin/bash

# merge the results to single CSV
# reset file if exists
rm -f all_${output_file}
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
				echo  ${output_dir}/${output_file} 
				if ! [ -f ${output_dir}/${output_file} ]; then
					continue
				fi
				if ! [ -f all_${output_file} ] ; then
					head -n1 ${output_dir}/${output_file} > all_${output_file}	 
				fi	
				echo "merging ${output_dir}/${output_file}"
				tail -n+2 ${output_dir}/${output_file} >> all_${output_file}	
			done		
		done
	done
done 


# analyze the results
echo "Analyzing..."
output_dir=$(echo ${output_file} | awk -F $'.' '{print $1}')_analysis
python -u ~/antibody_sequence_embedding/executable_scripts/analyze_classification_results.py --input_file all_${output_file} --output_dir ${output_dir} > analyze_classification_results.log.txt
