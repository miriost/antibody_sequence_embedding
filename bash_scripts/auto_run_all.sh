#!/bin/bash

Input_file=/media/miri-o/Documents/files_to_vectorise/HCV_sampled_50k.csv
model_file=/media/miri-o/Documents/Immune2vec/trained_models/Celiac_n_3_trimming_2_1.model
log_file=/media/miri-o/Documents/logs/auto_run.txt

cd /media/miri-o/Documents/miris_tools/executable_scripts/

echo "1. Generating vectors"
python generate_vectors.py -c "JUNC_AA" $Input_file $model_file > $log_file
line=$(tail -n 1 $log_file)
full_file_path=${line##*:}
filename=${line##*/*/}
only_path=${full_file_path/$filename}
filename=${filename/.csv}
data_file=$(tail -n 2 $log_file| head -n 1)
data_file=${data_file##*:}

n=35
m='PCA'
echo "2. Reducing dimensions"
python reduce_dimensions.py -i $full_file_path -o ${only_path}${filename}_after_${m}_${n}D.csv -n $n -m $m >> $log_file

line=$(tail -n 1 $log_file)
full_file_path=${line##*:}

n=10
d=2
TH=98
l='SUBJECT'
echo "3. Performing clustering"
python perform_clustering.py -m "kmeans" -n $n -d $d -TH $TH -l $l $full_file_path $data_file >> $log_file

line=$(tail -n 1 $log_file)
full_file_path=${line##*:}
dict_file='../../HCV_dictionary.csv'
python add_labels_to_feature_table.py $full_file_path $dict_file -l label -o observation >> $log_file

line=$(tail -n 1 $log_file)
full_file_path=${line##*:}

echo "4. Classifying"
python classify.py $full_file_path --labels_col_name 'labels' -M "logistic_regression" >>  $log_file

