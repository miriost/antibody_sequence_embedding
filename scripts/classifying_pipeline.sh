#!/bin/bash

trap "exit" INT


#!/bin/bash

trap "exit" INT

function show_help {
  echo "Build clusters and feature tables for train/test folds."
  echo "--description DESCRIPTION - Mandatory, decription for the calssification results file name"
  echo "--folds FOLDS - Optional, space separated list of folds numbers. Deafult is 0."
  echo "--model MODEL - Optional, which model to use for the classification. Default is logistic_regression."
  echo "--optimize OPTMIZE - Optional, try to optimize the training using cross validation. Default is False."
  echo "--knn KNN - Optional, space separated list of the K nearest neighbors to serach in the clusters construction. Deafult is 100."
  echo "--max_distance_percentil MAX_DISTANCE_percentil - Optional, space separated list of max distance pecentile for filtering cluster neighbors. Default is \"100\" (all knn neighbors)."
  echo "--min_significance MIN_significance - Optional, space separated list of minimal significance threshould for the cluster selection. Default is \"0.7\"."
  echo "--min_subjects MIN_SUBJECTS - Optional, a space separated list of the number minimal of subjects threshold for the cluster selection. Default is \"7\"."
  echo "--work_dir WORK_DIR - Optional, the folds root directory where the folds are. Default is \"./\"."
}

folds=0
knn=100
min_significance="0.7"
work_dir=./
min_subjects=7
max_distance_percentil=100
description=""
model=logistic_regression
optimize=False

# Read command line options
ARGUMENT_LIST=(
    "help"
    "description"
    "folds"
    "knn"
    "max_distance_percentil"
    "min_significance"
    "min_subjects"
    "work_dir"
    "model"
)

# read arguments
opts=$(getopt \
    --longoptions "$(printf "%s:," "${ARGUMENT_LIST[@]}")" \
    --name "$(basename "$0")" \
    --options "" \
    -- "$@"
)

eval set --$opts

while [[ $# -gt 0 ]]; do
    case "$1" in
      --help)
        show_help
        exit 0
        ;;
      --description)
        description=$2
        shift 2
        ;;
      --folds)
        folds=$2
        shift 2
        ;;
      --knn)
        knn=$2
        shift 2
        ;;
      --max_distance_percentil)
        max_distance_percentil=$2
        shift 2
        ;;
      --min_significance)
        min_significance=$2
        shift 2
        ;;
      --min_subjects)
        min_subjects=$2
        shift 2
        ;;
      --work_dir)
        work_dir=$2
        shift 2
        ;;
      --model)
        model=$2
        shift 2
        ;;
      --optmize)
        optmize=$2
        shift 2
        ;;
      *)
        break
        ;;
    esac
done

# arguments validation
if [ -z "${description}" ]; then
	echo "Missing mandatory description argument. Exiting..."
	show_help
	echo "pre_clustering_pipeline.sh -h for additional help."
	exit -1
fi

# change to the working directory
echo "cd ${work_dir}"
cd ${work_dir}

data_file=${description}_FILTERED_TRAIN.tsv
vectors_file=${description}_VECTORS_TRAIN.npy

test_data_file=${description}_FILTERED_TEST.tsv
test_vectors_file=${description}_VECTORS_TEST.npy

${output_file}=${description}.tsv

# loop folds
for fold in ${folds} ; do
	echo "Fold ${fold}"; echo ""
	fold_dir=FOLD${fold}

	# loop knn value
	for knn_itr in ${knn}; do
		echo "knn ${knn_itr}"; echo ""
		knn_dir=${fold_dir}/knn_${knn_itr}

		#loop max features
		for min_subjects_itr in ${min_subjects}; do
			echo "min_subjects ${min_subjects_itr}"; echo ""
			min_subjects_dir=${knn_dir}/min_subjects_${min_subjects_itr}

      #loop min significance
      for min_significance_itr in ${min_significance}; do
        echo "min_significance ${min_significance_itr}"; echo ""
        min_significance_dir=${min_subjects_dir}/min_significance_${min_significance_itr}

        #loop max_distance_percentil
        for max_distance_percentil_itr in ${max_distance_percentil}; do
          echo "max_distance_percentil ${max_distance_percentil_itr}"; echo ""
          max_distance_percentil_dir=${min_significance_dir}/max_distance_percentil_${max_distance_percentil_itr}

          cmd="nice -19 python -u ~/antibody_sequence_embedding/classify_repertoire.py --train_file ${max_distance_percentil_dir}/train_feature_table.csv --test_file
          ${max_distance_percentil_dir}/test_feature_table.csv --col_names=\"knn,min_subjects,min_significance,max_distance_percentil\" --col_values=\"${knn_itr},${min_subjects_itr},
          ${min_significance_itr},${max_distance_percentil_itr}\" --output_file ${output_dir}/${output_file} --models ${models} --grid_search=${optimize}"
          echo ${cmd}
          eval ${cmd}

        done # max_distance_percentil loop
      done # min_significance loop
		done # min_subjects loop
	done # knnloop
done # fold loop

rm -f all_${output_file}
# loop folds
for fold in ${folds} ; do
	echo "Fold ${fold}"; echo ""
	fold_dir=FOLD${fold}

	# loop knn value
	for knn_itr in ${knn}; do
		knn_dir=${fold_dir}/knn_${knn_itr}

		#loop max features
		for min_subjects_itr in ${min_subjects}; do
			min_subjects_dir=${knn_dir}/min_subjects_${min_subjects_itr}

      #loop min significance
      for min_significance_itr in ${min_significance}; do
        min_significance_dir=${min_subjects_dir}/min_significance_${min_significance_itr}

        #loop max_distance_percentil
        for max_distance_percentil_itr in ${max_distance_percentil}; do
          max_distance_percentil_dir=${min_significance_dir}/max_distance_percentil_${max_distance_percentil_itr}

          if ! [ -f ${output_dir}/${output_file} ]; then
				    continue
			    fi

          if ! [ -f all_${output_file} ] ; then
            head -n1 ${output_dir}/${output_file} > all_${output_file}
          fi

          echo "merging ${output_dir}/${output_file}"
          tail -n+2 ${output_dir}/${output_file} >> all_${output_file}

        done # max_distance_percentil loop
      done # min_significance loop
		done # min_subjects loop
	done # knnloop
done # fold loop

# analyze the results
echo "Analyzing..."
output_dir=$(echo ${output_file} | awk -F $'.' '{print $1}')_analysis
cmd="nice -19 python -u ~/antibody_sequence_embedding/analyze_classification.py --input_file all_${output_file} --output_dir ${output_dir} 2>&1 | tee analyze_classification_results.log.txt"
echo ${cmd}
eval ${cmd}