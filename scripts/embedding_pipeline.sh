#!/bin/bash

trap "exit" INT

trim_start=2
trim_end=1
query=""
sample_size=2000
description=""
model_file=""
random_seed=0
work_dir=./
exclude_duplicates="True"
sampe_file="False"

function show_help {
  echo "Execute all the pre processing steps before clustering a data set."
  echo "--help - show this help message"
  echo "--description DESCRIPTION - Mandatory, decription of which will be used as suffix for the resulting files and added columns. For example \"-d celiac_heavy_chain\"."
  echo "--query QUERY - A KQL query to import data from the elastic search. Mandatory if <decsription>.tsv file doesn't exist in input direction"
  echo "--model_file MODEL_FILE - Optional, model file to be used to embed the junction_aa, if not provided a model will be generated from data set"
  echo "--min_sample_size SAMPLE_SIZE - Optional, number of sequences to sample from each subject repertoire for the clustering. Default is 2000"
  echo "--sample - Optional, whether to sampel min_sample_size from each subject, Default is False"
  echo "--random_seed RANDOM_SEED - Optional, seed for the sampling. Default is 0."
  echo "--n_folds N_FOLDS - Optional, number of folds for the repeated cross validation splitting. Deafult is 10."
  echo "--work_dir WORK_DIR - Optional, the work directory. Deafult is \"./\""
  echo "--naive NAIVE - Optional, Is the data naive cells - affects the squence filtering during the reprocess_mkdb. Deafult is True."
  echo "--exclude_duplicates EXCLUDE_DUPLICATES - Optional, exclude duplicated cdr3_aa sequences in teh subject when sampling and/or before embedding. Deafult is True."
}

# Read command line options
ARGUMENT_LIST=(
    "help"
    "description"
    "query"
    "model_file"
    "min_sample_size"
    "sample"
    "random_seed"
    "n_folds"
    "work_dir"
    "naive"
    "exclude_duplicates"
)

# read arguments
opts=$(getopt \
    --longoptions "$(printf "%s:," "${ARGUMENT_LIST[@]}")" \
    --name "$(basename "$0")" \
    --options "" \
    -- "$@"
)

echo $opts

eval set --$opts

while true; do
    case "$1" in
    --help)
        show_help
        exit 0
        ;;
    --description)
        shift
        description=$1
        ;;
    --query)
        shift
        query=$1
        ;;
    --model_file)
        shift
        model_file=$1
        ;;
    --min_sample_size)
        shift
        min_sample_size=$1
        ;;
    --sample)
        shift
        sample=$1
        ;;
    --random_seed)
        shift
        random_seed=$1
        ;;
    --n_folds)
        shift
        n_folds=$1
        ;;
    --work_dir)
        shift
        work_dir=$1
        ;;
    --naive)
        shift
        naive=$1
        ;;
    --)
      shift
      break
      ;;
    esac
    shift
done

# change dirctory to input directory
cd ${work_dir}

# arguments validation
if [ -z "${description}" ]; then
	echo "Missing mandatory description argument. Exiting..."
	print_help
	echo "pre_clustering_pipeline.sh -h for additional help."
	exit -1
fi

if [ -z "${model_file}" ]; then
	model_desc=${description}_${n_dim}dim
else
	model_desc=$(echo ${model_file} | awk -F $'.' '{print $1}')
fi

# import file from elastic search data base
import_file=${description}.tsv
if [ -f ${import_file} ]; then
	echo "File ${import_file} already exists, skipping data import."; echo ""
else
	if [ -z "${query}" ]; then
		echo "Missing mandatory query argument. Exiting..."
		print_help
		echo "pre_clustering_pipeline.sh -h for additional help."
		exit -1
	fi
	
	echo "Importing data..."
	nice -19 python -u /work/data/es_lab/import_data.py --index sequence_v1 --query "${query}" --include_fields --count_only False --output_file ${import_file} --max_records 4000000
fi

# Reprocess mkdb
mkdb_file=${description}_db.tsv
if [ -f ${mkdb_file} ]; then
  echo "File ${mkdb_file} already exists, skipping reprocess mkdb."; echo ""
else
  echo "Reprocess mkdb..."; echo ""
  nice -19 python -u ~/antibody_sequence_embedding/reprocess_makedb_dir.py ${import_file} ${desc}_db.tsv --naive ${naive} --min_seq_per_subject ${min_sample_size}
fi

if [[ "${sample_file}" == "True" ]]; then
  description=${description}_${model_desc}_sampled_n${sample_size}
fi

vectors_file=${description}_VECTORS.npy
data_file=${description}_FILTERED.tsv

if ! [ -f ${vectors_file} && -f ${data_file} ]; then
  echo "${vectors_file} and ${data_file} already exists, skipping generate model and generate vectors."; echo ""
else
  if ! [ -z "${model_file}" ]; then
    # model file was already provided
    if ! [ -f ${model_file} ]; then
      echo "Bad path for model file: ${model_file}. Exiting..."
      print_help
      echo "pre_clustering_pipeline.sh -h for additional help."
      exit -1
    else
      echo "Model file ${model_file} provided, skipping generate model."; echo ""
    fi
  else
    # generate model
    model_file=${model_desc}.model
    if [ -f ${model_file} ]; then
      echo "Model file ${model_file} already exists, skipping generate model."; echo ""
    else
      echo "Generate model..."; echo ""
      nice -19 python ~/antibody_sequence_embedding/generate_model.py ${mkdb_file} ${model_desc} --n_dim ${n_dim}
    fi
  fi

  # sample file
  if [[ "${sample_file}" == "True" ]]; then
    sampled_file=${description}.tsv
    if [ -f ${sampled_file} ]; then
      echo "File ${sampled_file} already exists, skipping sampling."; echo ""
    else
      echo "Sample file..."; echo ""
      nice -19 python ~/antibody_sequence_embedding/sample_file.py ${mkdb_file} ${sampled_file} ${min_sample_size} --exclude_duplicates ${exclude_duplicates}
    fi
    sampled_file=${mkdb_file}
  fi

  # generate vectors
  echo "Generate vectors..."; echo ""
  nice -19 python ~/antibody_sequence_embedding/generate_vectors.py ${sampled_file} ${model_file} --drop_duplicates ${exclude_duplicates}
fi


# create repeated cross validation folds
folds_dir=${description}_${n_folds}folds
if [ -d ${folds_dir} ]; then
	echo "Folds dir ${folds_dir} already exists, skipping folds creation."; echo ""
else
	echo "Split folds..."; echo ""
	echo ${sampled_file}
	nice -19 python ~/antibody_sequence_embedding/split_data_train_test_folds.py ${data_file} --balance_train_labels True --n_splits ${n_folds} --output_dir ${folds_dir}
	nice -19 python ~/antibody_sequence_embedding/split_data_train_test_folds.py ${data_file} ${vectors_file} --n_splits ${n_folds}
fi

