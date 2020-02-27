#!/bin/bash

Input_file=/media/miri-o/Documents/HCV_data_only_CI_SC_10K_per_subject.csv
model_file=/media/miri-o/Documents/trained_models_NEW/CDR3_from_HCV_trimmied_3_4.model.model
log_file=/media/miri-o/Documents/logs/auto_run_HCV_data_only_CI_SC_10K_per_subject_020120_3d.txt
score_file=/media/miri-o/Documents/logs/scores_data_only_CI_SC_10K_per_subject_020120_3d.txt

vector_file_path=/media/miri-o/Documents/vectors/HCV_data_only_CI_SC_10K_per_subject_CDR3_from_HCV_trimmied_3_4_VECTORS_tSNE_3D.csv

data_file=/media/miri-o/Documents/filtered_data_sets/HCV_data_only_CI_SC_10K_per_subject_CDR3_from_HCV_trimmied_3_4_FILTERED_DATA.csv
echo 'Inputfile,model_name,dimension_reduction_method,number_of_dimensions,clustering_method,number_of_clusters,diversity_TH,classification_method,score' > $score_file

cd /media/miri-o/Documents/miris_tools/executable_scripts/

#echo "Generating vectors"
#python generate_vectors.py -c "JUNC_AA_trimmed_3_4" $Input_file $model_file > $log_file
#line=$(tail -n 1 $log_file)
#vector_file_path=${line##*:}
#filename=${line##*/*/}
#only_path=${vector_file_path/$filename}
#filename=${filename/.csv}
#data_file=$(tail -n 2 $log_file| head -n 1)
#data_file=${data_file##*:}



# dim_method='PCA'
#for dims in {2..3}
#do
	#echo "reducing dimensions to" $dims
	#echo $vector_file_path
	#python reduce_dimensions.py -i $vector_file_path -o ${only_path}${filename}_after_${dim_method}_${dims}D.csv -n $dims -m $dim_method >> $log_file

	#line=$(tail -n 1 $log_file)
clustering_file=$vector_file_path
for n in {5..30}
do
	d=2
	TH=98
	l='SUBJECT'
	clust_method="kmeans"
	echo "Performing clustering, n = "$n
	python perform_clustering.py -m $clust_method -n $n -d $d -TH $TH -l $l $clustering_file $data_file >> $log_file

	line=$(tail -n 1 $log_file)
	labels_file=${line##*:}
	clusters=$(tail -n 2 $log_file | head -n 1)
	clusters=${clusters##*x }
	C=' col'
	clusters=${clusters%$C*}


	dict_file='../../HCV_dictionary.csv'
	python add_labels_to_feature_table.py $labels_file $dict_file -l label -o observation >> $log_file

	line=$(tail -n 1 $log_file)
	classify_file=${line##*:}


	declare -a class_method=('LR' 'DT' 'kNN' 'LSVM' 'RBF_SVM' 'Gaussian' 'RF' 'MLP' 'ADA' 'NB')
	for method in "${class_method[@]}"
	do
		echo "Classifying" "$method"
		python classify.py $classify_file --labels_col_name 'labels' -M $method >>  $log_file
		line=$(tail -n 1 $log_file)
		score=${line##*: }
		echo ${Input_file##*/*/},${model_file##*/*/},${dim_method},${dims},${clust_method},${clusters},${TH},${method},${score} >> $score_file
	done
done
