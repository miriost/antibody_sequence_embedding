#!/bin/bash
cd /media/miri-o/Documents/miris_tools/executable_scripts/

n=398
echo $n
echo "~~~ logistic_regression features 1 to $n," 
python /media/miri-o/Documents/miris_tools/executable_scripts/classify.py /media/miri-o/Documents/feature_table/HCV_CDR3_CI_AND_SC_Celiac_n_3_trimming_2_1_VECTORS_after_PCA_5D_feature_table_with_labels.csv --feature_cols 1 $n --labels_col_name 'labels' -M "logistic_regression"

echo "~~~ decision tree features 1 to $n," 
python /media/miri-o/Documents/miris_tools/executable_scripts/classify.py /media/miri-o/Documents/feature_table/HCV_CDR3_CI_AND_SC_Celiac_n_3_trimming_2_1_VECTORS_after_PCA_5D_feature_table_with_labels.csv --feature_cols 1 $n --labels_col_name 'labels' -M "decision_tree" 

echo "~~~ regulrized logistic regression, features 1 to $n," 
python /media/miri-o/Documents/miris_tools/executable_scripts/classify.py /media/miri-o/Documents/feature_table/HCV_CDR3_CI_AND_SC_Celiac_n_3_trimming_2_1_VECTORS_after_PCA_5D_feature_table_with_labels.csv --feature_cols 1 $n --labels_col_name 'labels' -M "RLR" 
	

