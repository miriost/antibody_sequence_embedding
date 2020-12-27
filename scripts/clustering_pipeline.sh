#!/bin/bash

trap "exit" INT

function show_help {
  echo "Build clusters and feature tables for train/test folds."
  echo "--folds FOLDS - Optional, space separated list of folds numbers. Deafult is 0."
  echo "--knn KNN - Optional, space separated list of the K nearest neighbors to serach in the clusters construction. Deafult is 100."
  echo "--max_distance MAX_DISTANCE - Optional, space separated list of max distance cutoff for filtering cluster neighbors. Default is \"100\" (all knn neighbors)."
  echo "--min_significance MIN_significance - Optional, space separated list of minimal significance threshould for the cluster selection. Default is \"0.7\"."
  echo "--min_subjects MIN_SUBJECTS - Optional, a space separated list of the number minimal of subjects threshold for the cluster selection. Default is \"7\"."
  echo "--work_dir WORK_DIR - Optional, the folds root directory where the folds are. Default is \"./\"."
  echo "--same_gene SAME_GENE - Optional, Search knn enforcing same vgene and jgene, default is False"
  echo "--same_junction_len  - Optional, Search knn enforcing same cdr3 length, default is False"
  echo "--dist_metric - Optional. which dist_metric to use for the custering, default is \"euclidean\" "
  echo "--do_clustering - Optional. Use hierarchical clustering and not full linkage clustering. Deafulat is True"
  echo "--thread_memory - Optional. Thread memory argument for ray.init()."
}

folds=0
knn=100
min_significance="0.7"
work_dir=./
min_subjects=7
max_distance=100
description=""
same_gene=False
same_junction_len=False
dist_metric='euclidean'
do_clustering=True
thread_memory=0

# Read command line options
ARGUMENT_LIST=(
    "help"
    "description"
    "folds"
    "knn"
    "max_distance"
    "min_significance"
    "min_subjects"
    "work_dir"
    "same_gene"
    "same_junction_len"
    "dist_metric"
    "do_clustering"
    "thread_memory"
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
      --max_distance)
        max_distance=$2
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
      --same_gene)
        same_gene=$2
        shift 2
        ;;
      --same_junction_len)
        same_junction_len=$2
        shift 2
        ;;
      --dist_metric)
        dist_metric=$2
        shift 2
        ;;
      --do_clustering)
        do_clustering=$2
        shift 2
        ;;
      --thread_memory)
        thread_memory=$2
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

# loop folds
for fold in ${folds} ; do
	echo "Fold ${fold}"; echo ""
	fold_dir=FOLD${fold}
	
	# loop knn value
	for knn_itr in ${knn}; do
		echo "knn ${knn_itr}"; echo ""
		knn_dir=${fold_dir}/knn_${knn_itr}
		mkdir -p ${knn_dir}

		if [ -f ${knn_dir}/${knn_itr}knn_neighbors.npy ] && [ -f ${knn_dir}/${knn_itr}knn_distances.npy ] && ![[ "${do_clustering}" == "True" ]] ; then
			echo "${knn_dir}/${knn_itr}knn_neighbors.npy and ${knn_dir}/${knn_itr}knn_distances.npy already exists, skipping KNN search."
		else
			# search K nearest neighbors
			echo "Starting KNN search..."
			cmd="nice -19 python -u ~/antibody_sequence_embedding/build_cluster_proximity.py ${fold_dir}/${data_file} ${fold_dir}/${vectors_file} ${knn_itr}knn ${knn_dir} --cluster_size ${knn_itr}
			--num_cpus 12 --same_gene ${same_gene} --same_junction_len ${same_junction_len} --dist_metric ${dist_metric} --do_clustering ${do_clustering} --cluster_id_column ${dist_metric}_cluster_id
			--thread_memory ${thread_memory}"
			echo ${cmd}
			eval ${cmd}
		fi

		if [[ "${do_clustering}" == "True" ]]; then
		  knn_dir=${fold_dir}
		fi

		#loop max features
		for min_subjects_itr in ${min_subjects}; do
			echo "min_subjects ${min_subjects_itr}"; echo ""
			min_subjects_dir=${knn_dir}/min_subjects_${min_subjects_itr}
			mkdir -p ${min_subjects_dir}

      #loop min significance
      for min_significance_itr in ${min_significance}; do
        echo "min_significance ${min_significance_itr}"; echo ""
        min_significance_dir=${min_subjects_dir}/min_significance_${min_significance_itr}
        mkdir -p ${min_significance_dir}

        #loop max_distance
        for max_distance_itr in ${max_distance}; do
          echo "max_distance ${max_distance_itr}"; echo ""
          max_distance_dir=${min_significance_dir}/max_distance_${max_distance_itr}
          mkdir -p ${max_distance_dir}

          if [ -f ${max_distance_dir}/feature_list.tsv ] ; then
            echo "${max_distance_dir}/feature_list.tsv already exists, skipping building feature list."
          else
            # build feature list
            if [[ "${do_clustering}" == "True" ]]; then
              if [[ "${dist_metric}" == "manhattan" ]]; then
                is_manhattan=True
              else
                is_manhattan=False
              fi
              cmd="nice -19 python -u ~/antibody_sequence_embedding/filter_clusters.py ${fold_dir}/${data_file} ${fold_dir}/${vectors_file} ${knn_dir}/${knn_itr}knn_distances.npy
              ${knn_dir}/${knn_itr}knn_neighbors.npy ${max_distance_dir} feature_list ${dist_metric}_cluster_id --subjects_th ${min_subjects_itr} --significance_th ${min_significance_itr}
              --max_distance_th ${max_distance_itr} --is_manhattan${is_manhattan}"
            else
              cmd="nice -19 python -u ~/antibody_sequence_embedding/build_knn_cluster_proximity_feature_list.py ${fold_dir}/${data_file} ${fold_dir}/${vectors_file}
              ${knn_dir}/${knn_itr}knn_distances.npy ${knn_dir}/${knn_itr}knn_neighbors.npy ${max_distance_dir} feature_list --min_subjects ${min_subjects_itr} --min_significance
              ${min_significance_itr} --max_distance ${max_distance_itr} --num_cpus 12 --dist_metric ${dist_metric} --same_gene ${same_gene} --same_junction_len ${same_junction_len}"
            fi
            echo ${cmd}
            eval ${cmd}
          fi

          if [ -f ${max_distance_dir}/train_feature_table.csv ] ; then
            echo "${max_distance_dir}/train_feature_table.csv already exists, skipping building train feature table."
          else
            cmd="nice -19 python ~/antibody_sequence_embedding/build_cluster_proximity_feature_table.py ${fold_dir}/${data_file} ${fold_dir}/${vectors_file}
            ${max_distance_dir}/feature_list.tsv train ${max_distance_dir} --dist_metric ${dist_metric} --same_gene ${same_gene} --same_junction_len ${same_junction_len}"
            echo ${cmd}
            eval ${cmd}
          fi

          if [ -f ${max_distance_dir}/test_feature_table.csv ] ; then
            echo "${max_distance_dir}/test_feature_table.csv already exists, skipping building train feature table."
          else
            cmd="nice -19 python ~/antibody_sequence_embedding/build_cluster_proximity_feature_table.py ${fold_dir}/${test_data_file} ${fold_dir}/${test_vectors_file}
            ${max_distance_dir}/feature_list.tsv test ${max_distance_dir} --dist_metric ${dist_metric}"
            echo ${cmd}
            eval ${cmd}
          fi

          if [ -f ${max_distance_dir}/selected_train_feature_table.csv ] && [ -f ${max_distance_dir}/selected_test_feature_table.csv ] ; then
            echo "${max_distance_dir}/selected_train_feature_table.csv and ${max_distance_dir}/selected_test_feature_table.csv already exits, skipping feature selection."
          else
            cmd="nice -19 python ~/antibody_sequence_embedding/select_features.py ${max_distance_dir}/train_feature_table.csv ${max_distance_dir}/test_feature_table.csv"
            echo ${cmd}
            eval ${cmd}
          fi

        done # max_distance loop
      done # min_significance loop
		done # min_subjects loop
	done # knnloop
done # fold loop
