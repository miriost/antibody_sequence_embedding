#!/bin/bash
cd /media/miri-o/Documents/miris_tools/executable_scripts/

for n in {2..398}
do
	echo $n
	echo "~~~ logistic_regression features 1 to $n," >> logistic_regression.txt
	python /media/miri-o/Documents/miris_tools/executable_scripts/classify.py /media/miri-o/Documents/feature_table/HCV_sampled_50k_Celiac_n_3_trimming_2_1_VECTORS_after_TSNE_2D_feature_table_with_labels.csv --feature_cols 1 $n --labels_col_name 'labels' -M "logistic_regression" | tail -n 4 >>  logistic_regression.txt

	echo "~~~ decision tree features 1 to $n," >> decision_tree.txt
	python /media/miri-o/Documents/miris_tools/executable_scripts/classify.py /media/miri-o/Documents/feature_table/HCV_sampled_50k_Celiac_n_3_trimming_2_1_VECTORS_after_TSNE_2D_feature_table_with_labels.csv --feature_cols 1 $n --labels_col_name 'labels' -M "decision_tree" | tail -n 4 >> decision_tree.txt
	
done

