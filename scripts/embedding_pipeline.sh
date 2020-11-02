#!/bin/bash

trap "exit" INT

trim_start=2
trim_end=1
query=""
sample_size=10000
description=""
model_file=""
random_seed=0
work_dir=./
usage="Usage - pre_clustering_pipeline.sh -d <description> -q [query] -s [sample_size] -t [trim_start:trim_end] -m [model_file] -r [random_seed] -n[num of folds] -w [work_dir]"
help="Execute all the pre processing steps before clustring a data set.
-d DESCRIPTION - Mandatory, decription of which will be used as suffix for the resulting files and added columns. For example \"-d celiac_heavy_chain\".
-q QUERY - A KQL query to import data from the elastic search. Mandatory if <decsription>.tsv file doesn't exist in input direction
-t TRIM_START:TRIM_END - Optional, Amino Acids to trim from the start of the junction_aa. trim_start - Amino Acids to trim from the end for the junction_add. Default is 1.
-m MODEL_FILE - Optional, model file to be used to embed the junction_aa, if not provided a model will be generated from data set
-s SAMPLE_SIZE - Optional, number of sequences to sample from each subject repertoire for the clustering. Default is 10000
-r RANDOM_SEED - Optional, seed for the sampling. Default is 0.
-n N_FOLDS - Optional, number of folds for the repeated cross validation splitting. Deafult is 10.
-w WORK_DIR - Optional, the work directory. Deafult is \"./\"
"

while getopts "hq:s:t:d:m:r:n:w:" opt; do
	case ${opt} in
		h ) echo "${usage}" ; echo "${help}"; exit 1
      			;;
    		q ) query=${OPTARG}
		     	;;
		s ) sample_size=${OPTARG}
			;;
		t ) trim_start=$(echo ${OPTARG} | awk -F $':' '{print $1}'); trim_end=$(echo ${OPTARG} | awk -F $':' '{print $2}')
			;;
		d ) description=${OPTARG}
			;;
		m ) model_file=${OPTARG}
			;;
		r ) random_seed=${OPTARG}
			;;
		n ) n_folds=${OPTARG}
			;;
		w ) input_dir=${OPTARG}
			;;
		\? ) echo "${usage}"; exit 1
      			;;
	esac
done

# change dirctory to input directory
cd ${work_dir}

# arguments validation
if [ -z "${description}" ]; then
	echo "Missing mandatory description argument. Exiting..."
	echo "${usage}"
	echo "pre_clustering_pipeline.sh -h for additional help."
	exit -1
fi

if [ -z "${model_file}" ]; then
	model_desc=${description}_trim_${trim_start}_${trim_end}_prot2vec
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
		echo "${usage}"
		echo "pre_clustering_pipeline.sh -h for additional help."
		exit -1
	fi
	
	echo "Importing data..."
	python /work/data/es_lab/import_data.py --index sequence_v1 --query "${query}" --include_fields "junction;junction_length;productive;consensus_count;junction_aa;v_call;${model_desc};repertoire.subject_id;repertoire.repertoire_name;repertoire.disease_diagnosis" --count_only False --output_file ${import_file} --max_records 4000000
fi

# junction might be already embedded
if ! [ -z "$(head -n1 ${import_file} | grep ${model_desc})" ]; then
	echo "Column ${model_desc} already exists in ${import_file}, skipping reprocess mkdb, generate model and generate vectors."	
	mkdb_file=${import_file}
else
	# Reprocess mkdb
	mkdb_file=${description}_trim_${trim_start}_${trim_end}.tsv
	if [ -f ${mkdb_file} ]; then
		echo "File ${mkdb_file} already exists, skipping reprocess mkdb."; echo ""
	else
		echo "Reprocess mkdb..."; echo ""
		python ~/antibody_sequence_embedding/executable_scripts/reprocess_makedb_dir.py --input_files ${import_file} --output_file ${mkdb_file} --trim 2 1 --min_seq_per_subject ${sample_size}
	fi
	
	if ! [ -z "$(head -n1 ${mkdb_file} | grep ${model_desc})" ]; then
		echo "Column ${model_desc} already exists in ${mkdb_file}, skipping generate model and generate vectors."; echo ""
	else
		if ! [ -z "${model_file}" ]; then
			# model file was already provided
			if ! [ -f ${model_file} ]; then
				echo "Bad path for model file: ${model_file}. Exiting..."
				echo "${usage}"
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
				python ~/antibody_sequence_embedding/executable_scripts/generate_model.py --data_file ${mkdb_file} --data_column junction_aa_trim_${trim_start}_${trim_end} --desc ${model_desc} --n_gram 3
			fi
		fi
		
		# generate vectors
		echo "Generate vectors..."; echo ""	
		python ~/antibody_sequence_embedding/executable_scripts/generate_vectors.py ${model_file} --input_file ${mkdb_file} --column junction_aa_trim_${trim_start}_${trim_end} --output_column ${model_desc} --inline True --size 100
	fi
fi

# sample file
sampled_file=${description}_trim_${trim_start}_${trim_end}_model_${model_desc}_sampled_n${sample_size}_seed${random_seed}.tsv
if [ -f ${sampled_file} ]; then
	echo "File ${sampled_file} already exists, skipping sampling."; echo ""
else
	echo "Sample file..."; echo ""
	python ~/antibody_sequence_embedding/executable_scripts/sample_file.py --input_data_file ${mkdb_file} --exclude_dup_column junction_aa_trim_${trim_start}_${trim_end} --min_samples ${sample_size} --output_data_file ${sampled_file}
fi

# create repeated cross validation folds
folds_dir=${description}_trim_${trim_start}_${trim_end}_model_${model_desc}_sampled_n${sample_size}_seed${random_seed}_${n_folds}folds
if [ -d ${folds_dir} ]; then
	echo "Folds dir ${folds_dir} already exists, skipping folds creation."; echo ""
else
	echo "Split folds..."; echo ""
	echo ${sampled_file}
	python ~/antibody_sequence_embedding/executable_scripts/split_data_train_test_folds.py ${sampled_file} --test_fraction "0.1" --repeated=True --output_dir ${folds_dir} --number_of_folds ${n_folds}
fi

