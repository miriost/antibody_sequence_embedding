#!/bin/bash

USAGE="usage run_pipeline.sh <data_folder> <data_name> <min_step=0(reporcess)/1(generate_model)/2(generate_vectores)/3(generate_labels)/4(reduce_dimensions)/5(vfamily_analysis)] <max_step>";


DIR=$1
if ! [ -d $DIR ]; then
	echo $USAGE
	exit -1
fi
echo "data_dir = $DIR"

NAME="$2"
if [ "$NAME" == "" ]; then
	echo $USAGE
	exit -1
fi
echo "data_name = $NAME"

MIN_STEP=$3
if ! [ -n "$MIN_STEP" ] || ! [ "$MIN_STEP" -eq "$MIN_STEP" ]; then
	echo $USAGE
	exit -1
fi
echo "min_step = $MIN_STEP"

MAX_STEP=$4
if ! [ -n "$MAX_STEP" ] || ! [ "$MAX_STEP" -eq "$MAX_STEP" ] || [ $MAX_STEP -lt $MIN_STEP ]; then
	echo $USAGE
	exit -1
fi
echo "max_step $MAX_STEP"

mkdir -p ${DIR}/logs/

if [ $MIN_STEP -le 0 ] && [ 0 -le $MAX_STEP ]; then
	rm -f ${DIR}/logs/reprocess_makedb_dir.log.txt ;
	rm -f ${DIR}/logs/reprocess_makedb_dir.csv ;

	#python -u ~/antibody_sequence_embedding/reprocess_makedb_dir.py ${DIR}/data ${DIR}/data/${NAME}_aa_trim_3_4.csv --trim 3 4 2>&1 | tee ${DIR}/logs/reprocess_makedb_dir.log.txt ;
	python -u ~/antibody_sequence_embedding/reprocess_makedb_dir.py ${DIR}/data ${DIR}/data/${NAME}_aa_trim_3_4.csv --trim 0 4 2>&1 | tee ${DIR}/logs/reprocess_makedb_dir.log.txt ;
	cat ${DIR}/logs/reprocess_makedb_dir.log.txt | cut -d : -f 2 | grep -E "([0-9]+[ ]*$)|(.*.tab$)" | sed -z 's/\n/,/g' | sed -z 's/\([^,]\+.tab\)/\n\1/g' | sed 's/,$//g' >> ${DIR}/logs/reprocess_makedb_dir.csv ;
	# add columns names
	echo "$(echo 'file,total,functional,len_ge_12,CONSCOUNT_gt_1,no_N_or_gaps' | cat - reprocess_makedb_dir.csv)" > ${DIR}/logs/reprocess_makedb_dir.csv ;
	# discard empty lines
	cat ${DIR}/logs/reprocess_makedb_dir.csv | sed -r '/^\s*$/d' | tee ${DIR}/logs/reprocess_makedb_dir.csv ;
	python -u ~/antibody_sequence_embedding/db_to_cdr3_fasta.py -i ${DIR}/data/${NAME}_aa_trim_3_4.csv -o ${DIR}/${NAME}_cdr3_aa_trim_3_4.fasta -c JUNC_AA ; 
fi

if [ $MIN_STEP -le 1 ] && [ 1 -le $MAX_STEP ]; then
	rm -f ${DIR}/logs/generate_model.log.txt ; 
	mkdir -p ${DIR}/models/ ;
	#for p in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do 
	for p in 1.0; do
		pid = python -u ~/antibody_sequence_embedding/executable_scripts/generate_model.py -f ${DIR}/${NAME}_cdr3_aa_trim_3_4.fasta -o ${DIR}/models/${NAME}_${p} -p ${p} 2>&1 | tee -a ${DIR}/logs/generate_model.log.txt &  
	done ;
fi

if [ $MIN_STEP -le 2 ] && [ 2 -le $MAX_STEP ]; then
	mkdir -p ${DIR}/vectors/ ;
	mkdir -p ${DIR}/filtered_data_sets/ ;
	rm -f ${DIR}/logs/generate_vectors.log.txt ;
	for model in ${DIR}/models/*.model ; do 
		python -u ~/antibody_sequence_embedding/executable_scripts/generate_vectors.py ${DIR}/data/${NAME}_aa_trim_3_4.csv -o ${DIR} -c JUNC_AA ${model} 2>&1 | tee -a ${DIR}/logs/generate_vectors.log.txt & 
	done ;
fi

if [ $MIN_STEP -le 3 ] && [ 3 -le $MAX_STEP ]; then
	mkdir -p ${DIR}/vfamily_labels ;
	for vector_file in $(ls ${DIR}/filtered_data_sets/); do 
		Rscript ~/antibody_sequence_embedding/V_family_labels.R ${DIR}/filtered_data_sets/${vector_file} ${DIR}/vfamily_labels/$(echo "${vector_file}" '\t' | sed 's/FILTERED_DATA.csv/labels.csv/g') &
	done ;
fi

if [ $MIN_STEP -le 4 ] && [ 4 -le $MAX_STEP ]; then
	mkdir -p ${DIR}/vectors_reduced_to_10dim ;
	rm -f ${DIR}/logs/reduce_dimensions.log.txt ;
	for vector_file in $(ls ${DIR}/vectors/) ; do 
		python -u ~/antibody_sequence_embedding/executable_scripts/reduce_dimensions.py -i ${DIR}/vectors/${vector_file} -o ${DIR}/vectors_reduced_to_10dim/${vector_file} -n 10 -m PCA 2>&1 | tee -a ${DIR}/logs/reduce_dimensions.log.txt & 
	done ;
fi

if [ $MIN_STEP -le 5 ] && [ 5 -le $MAX_STEP ]; then
	rm -f ${DIR}/logs/V_family_analysis.log.txt ;
	for vector_file in $(ls ${DIR}/vectors_reduced_to_10dim/) ; do 
		python -u ~/antibody_sequence_embedding/V_family_analysis.py ${DIR}/vectors_reduced_to_10dim/${vector_file} ${DIR}/vfamily_labels/$(echo "${vector_file}" | sed 's/VECTORS.csv/labels.csv/g') -l V_FAMILY 2>&1 | tee -a ${DIR}/logs/V_family_analysis.log.txt & 
	done ;
fi


