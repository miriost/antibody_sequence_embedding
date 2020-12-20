#!/bin/bash

trap "exit" INT

function show_help {
  echo "Build clusters and feature tables for train/test folds."
  echo "folds FOLDS - Optional, space separated list of folds numbers. Deafult is 0."
  echo "knn KNN - Optional, space separated list of the K nearest neighbors to serach in the clusters construction. Deafult is 100."
  echo "max_distance_percentile MAX_DISTANCE_PERCENTILE - Optional, space separated list of max distance pecentile for filtering cluster neighbors. Default is \"100\" (all knn neighbors)."
  echo "min_significant MIN_SIGNIFICANT - Optional, space separated list of minimal significant threshould for the cluster selection. Default is \"0.7\"."
  echo "min_subjects MIN_SUBJECTS - Optional, a space separated list of the number minimal of subjects threshold for the cluster selection. Default is \"10\"."
  echo "work_dir WORK_DIR - Optional, the folds root directory where the folds are. Default is 50\"./\"."
}

folds=0
knn=100
significance_level="0.7"
work_dir=./
min_subjects=10
max_distance_percentile=100

# Read command line options
ARGUMENT_LIST=(
    "folds"
    "cluster_size"
    "max_distance_percentile"
    "min_significant"
    "min_subjects"
    "work_dir"
)

echo $opts

eval set --$opts

while true; do
    case "$1" in
    --help)
      show_help
      exit 0
      ;;
    --folds)
      shift
      folds=$1
      ;;
    --knn)
      shift
      knn=$1
      ;;
    --max_distance_percentile)
      shift
      max_distance_percentile=$1
      ;;
    --min_significant)
      shift
      min_significant=$1
      ;;
    --min_subjects)
      shift
      min_subjects=$1
      ;;
    --work_dir)
      shift
      work_dir=$1
      ;;
    --)
      shift
      break
      ;;
    esac
    shift
done

# change to the working directory
cd ${work_dir}

# loop folds
for fold in ${folds} ; do
	echo "Fold ${fold}"; echo ""
	fold_dir=FOLD${fold}
	
	# loop cluster size
	for knn_itr in ${knn}; do
		echo "knn ${knn_itr}"; echo ""
		knn_dir=${fold_dir}/knn_${knn_itr}
		mkdir -p ${knn_dir}

		if [ -f ${knn_dir}/${knn_itr}knn_neighbors.npy &&  -f ${knn_dir}/${knn_itr}knn_distances.npy ] ; then
			echo "${knn_dir}/${knn_itr}knn_neighbors.npy and ${knn_dir}/${knn_itr}knn_distances.npy already exists, skipping KNN search."
		else
			# search K nearest neighbors
			echo "Starting KNN search and analysis..."
			eval nice -19 python -u ~/antibody_sequence_embedding/executable_scripts/build_cluster_proximity.py ${fold_dir}/*_TRAIN.tsv ${fold_dir}/*_TRAIN.npy --cluster_size ${knn_itr} ${knn_itr}knn
			--output_folder_path ${knn_dir} --num_cpus 12
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
        done # max_distnace_percentile loop
      done # min_significant loop
		done # min_subjects loop
	done # knnloop
done # fold loop
