#!/bin/bash

if [ "$1" != "" ]; then
    Input_file=$1
else
    echo "Missing input file"
    exit 1
fi
file=${Input_file##*/*/}
file=${file/.csv}

model_file=/media/miri-o/Documents/Immune2vec/trained_models/Celiac_n_3_trimming_2_1.model
log_file=/media/miri-o/Documents/logs/auto_run_${file}_low_dimension.txt
score_file=/media/miri-o/Documents/logs/scores_${file}_low_dimension.txt
echo 'Inputfile,model_name,dimension_reduction_method,number_of_dimensions,clustering_method,number_of_clusters,diversity_TH,classification_method,score' > $score_file

cd /media/miri-o/Documents/miris_tools/executable_scripts/

echo "Generating vectors"
python generate_vectors.py -c "JUNC_AA" $Input_file $model_file > $log_file
line=$(tail -n 1 $log_file)
vector_file_path=${line##*:}
filename=${line##*/*/}
only_path=${vector_file_path/$filename}
filename=${filename/.csv}
data_file=$(tail -n 2 $log_file| head -n 1)
data_file=${data_file##*:}

dims_range=($(seq 2 1 10))
clust_range=($(seq 5 5 30))


dim_method='PCA'
for dims in ${dims_range[@]}
do
	echo "reducing dimensions to" $dims 
	echo $vector_file_path
	python reduce_dimensions.py -i $vector_file_path -o ${only_path}${filename}_after_${dim_method}_${dims}D.csv -n $dims -m $dim_method >> $log_file

	line=$(tail -n 1 $log_file)
	clustering_file=${line##*:}
	for n in ${clust_range[@]}
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

		 
		declare -a class_method=('LR' 'DT')
		for method in "${class_method[@]}"
		do
			echo "Classifying" "$method"
			python classify.py $classify_file --labels_col_name 'labels' -M $method >>  $log_file
			line=$(tail -n 1 $log_file)
			score=${line##*: }
			echo ${Input_file##*/*/},${model_file##*/*/},${dim_method},${dims},${clust_method},${clusters},${TH},${method},${score} >> $score_file
		done
	done
done
