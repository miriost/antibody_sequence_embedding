#!/bin/bash

trap "exit" INT

function show_help {
  echo "Build clusters and feature tables for train/test folds."
  echo "folds FOLDS - Optional, space separated list of folds numbers. Deafult is 0."
  echo "knn KNN - Optional, space separated list of the K nearest neighbors to serach in the clusters construction. Deafult is 100."
  echo "max_distance_percentile MAX_DISTANCE_PERCENTILE - Optional, space separated list of max distance pecentile for filtering cluster neighbors. Default is \"100\" (all knn neighbors)."
  echo "min_significant MIN_SIGNIFICANT - Optional, space separated list of minimal significant threshould for the cluster selection. Default is \"0.7\"."
  echo "min_subjects MIN_SUBJECTS - Optional, a space separated list of the number minimal of subjects threshold for the cluster selection. Default is \"7\"."
  echo "work_dir WORK_DIR - Optional, the folds root directory where the folds are. Default is \"./\"."
}

folds=0
knn=100
significance_level="0.7"
work_dir=./
min_subjects=7
max_distance_percentile=100
description=""

# Read command line options
ARGUMENT_LIST=(
    "help"
    "description"
    "folds"
    "knn"
    "max_distance_percentile"
    "min_significant"
    "min_subjects"
    "work_dir"
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
      --max_distance_percentile)
        max_distance_percentile=$2
        shift 2
        ;;
      --min_significant)
        min_significant=$2
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

# loop folds
for fold in ${folds} ; do
	echo "Fold ${fold}"; echo ""
	fold_dir=FOLD${fold}
	
	# loop knn value
	for knn_itr in ${knn}; do
		echo "knn ${knn_itr}"; echo ""
		knn_dir=${fold_dir}/knn_${knn_itr}
		mkdir -p ${knn_dir}

		if [ -f ${knn_dir}/${knn_itr}knn_neighbors.npy ] && [ -f ${knn_dir}/${knn_itr}knn_distances.npy ] ; then
			echo "${knn_dir}/${knn_itr}knn_neighbors.npy and ${knn_dir}/${knn_itr}knn_distances.npy already exists, skipping KNN search."
		else
			# search K nearest neighbors
			echo "Starting KNN search..."
			echo "nice -19 python -u ~/antibody_sequence_embedding/build_cluster_proximity.py ${fold_dir}/${data_file} ${fold_dir}/${vectors_file} ${knn_itr}knn ${knn_dir} --cluster_size ${knn_itr} --num_cpus 12"
			nice -19 python -u ~/antibody_sequence_embedding/build_cluster_proximity.py ${fold_dir}/${data_file} ${fold_dir}/${vectors_file} ${knn_itr}knn ${knn_dir} --cluster_size ${knn_itr} --num_cpus 12
		fi

		#loop max features
		for min_subjects_itr in ${min_subjects}; do
			echo "min_subjects ${min_subjects_itr}"; echo ""
			min_subjects_dir=${knn_dir}/min_subjects_${min_subjects_itr}
			mkdir -p ${min_subjects_dir}

      #loop min significant
      for min_significant_itr in ${min_significant}; do
        echo "min_significant ${min_significant_itr}"; echo ""
        min_significant_dir=${min_subjects_dir}/min_significant_${min_significant_itr}
        mkdir -p ${min_significant_dir}

        #loop max_distnace_percentile
        for max_distnace_percentile_itr in ${max_distnace_percentile}; do
          echo "max_distnace_percentile ${max_distnace_percentile_itr}"; echo ""
          max_distnace_percentile_dir=${min_significant_dir}/max_distnace_percentile_${max_distnace_percentile_itr}
          mkdir -p ${max_distnace_percentile_dir}

          if [ -f ${max_distnace_percentile_dir}/festure_list.tsv ] ; then
            echo "${max_distnace_percentile_dir}/festure_list.tsv already exists, skipping building train feature table."
            continue
          fi

          # build feature list
          echo "nice -19 eval python -u ~/antibody_sequence_embedding/build_knn_cluster_proximity_feature_list.py ${fold_dir}/${data_file} ${fold_dir}/${vectors_file} ${knn_dir}/${knn_itr}knn_distances.npy
          ${knn_dir}/${knn_itr}knn_neighbors.npy ${max_distnace_percentile_dir}
          100knn_${p}p_feature_list --min_subjects ${min_subjects_itr} --min_significant ${min_significant_itr} --max_distance_percentile ${max_distnace_percentile_itr}"
          nice -19 eval python -u ~/antibody_sequence_embedding/build_knn_cluster_proximity_feature_list.py ${fold_dir}/${data_file} ${fold_dir}/${vectors_file} ${knn_dir}/${knn_itr}knn_distances.npy
          ${knn_dir}/${knn_itr}knn_neighbors.npy ${max_distnace_percentile_dir}
          100knn_${p}p_feature_list --min_subjects ${min_subjects_itr} --min_significant ${min_significant_itr} --max_distance_percentile ${max_distnace_percentile_itr}

        done # max_distnace_percentile loop
      done # min_significant loop
		done # min_subjects loop
	done # knnloop
done # fold loop
