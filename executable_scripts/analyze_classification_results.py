import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns

def group_by_metric(df: pd.DataFrame, metric_list: list) -> pd.DataFrame:

	if len(metric_list) == 0:
		desc = df.loc[:, ['accuracy', 'f1_score', 'precision', 'recall']].describe()
		res = pd.DataFrame(index = [0])
		for idx, row in desc.iterrows():
			res[row.add_suffix('_' + idx).index.values.tolist()] = pd.DataFrame([row.values.tolist()], index=res.index)
		res['n_features_mean'] = df['n_features'].mean()
		res['n_folds'] = len(df['fold'].unique())
		return res

	metric = metric_list[0]
	metric_list = metric_list[1:]
	grouped_df = df.groupby(metric)
	df_list = []
	for name, group in grouped_df:
		tmp = group_by_metric(group, metric_list)
		tmp[metric] = name
		df_list += [tmp]

	res = pd.concat(df_list, ignore_index=True)

	return res


def main():
	parser = argparse.ArgumentParser(description='creates reports and graphs based on classification response csv file')

	parser.add_argument('--input_file', help='input classification results csv file')
	parser.add_argument('--group_by', help='comma separated list of columns based on do the analysis', type=str,
	                    default="cluster_size,significance,min_subj,model,report,key")
	parser.add_argument('--min_folds', help='min number of folds for analysis and ranking. By default will be the '
	                                        'max number of folds', type=str)

	args = parser.parse_args()

	if args.input_file is None or not (os.path.isfile(args.input_file)):
		print("missing or bad input_file argument")
		exit(1)

	input_file = pd.read_csv(args.input_file)

	if args.group_by is None:
		print("missing or bad input_file argument")
		exit(1)

	group_by_columns = args.group_by.split(',')
	for column in group_by_columns:
		if column is not in input_file.columns:
			print("{} is not in input_file columns".format(column))
			exit(1)

	min_folds = parser.min_folds
	if min_folds is None:
		min_folds = len(input_file['fold'].unique())

	df = group_by_metric(input_file, group_by_columns)

	# filter only test reports
	df_filtered = df.loc[(df['report'] == 'test') & (df['n_folds'] >= min_folds) & (df['key'] == 'macro avg'), : ]

	# print best 3 results in terms of avg_f1_score
	df_filtered = df_filtered.sort_values(by='f1_score_mean', ascending=False);
	print(df_filtered.head(3).transpose())

	rbf_svm_df = df_filtered.loc[df_filtered['model'] == 'RBF_SVM', :]
	sns.lineplot(data=rbf_svm_df, x="n_features_mean", y="f1_score_mean")
	sns.lineplot(data=rbf_svm_df, x="cluster_size", y="f1_score_mean")
	sns.lineplot(data=rbf_svm_df, x="min_subj", y="f1_score_mean")
	sns.lineplot(data=rbf_svm_df, x="significance", y="f1_score_mean")

indexing = (input_file['model'] == "ADA") & (input_file['significance'] == 0.6) & (input_file['cluster_size'] == 100) & (input_file['min_subj'] == 9)
input_file.loc[indexing, :]