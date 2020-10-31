import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json

def group_by_metric(df: pd.DataFrame, metric_list: list) -> pd.DataFrame:

	if len(metric_list) == 0:
		desc = df.loc[:, ['accuracy', 'f1_score', 'precision', 'recall']].describe()
		res = pd.DataFrame(index=[0], columns=['confusion_matrix'])
		for idx, row in desc.iterrows():
			res[row.add_suffix('_' + idx).index.values.tolist()] = pd.DataFrame([row.values.tolist()], index=res.index)
		res['n_features_mean'] = round(df['n_features'].mean())
		res['n_folds'] = len(df['fold'].unique())
		res['labels'] = df.loc[0, 'labels']
		res['parameters'] = format(df['parameters'].value_counts().to_dict())
		if 'confusion_matrix' in df.columns:
			arr = res.loc['confusion_matrix'].apply(lambda x: np.array(json.loads(x)))
			res.at[0, 'confusion_matrix'] = sum(arr.to_list()).to_list()

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
	parser.add_argument('--processed_input_file', help='already processed classification results csv file')
	parser.add_argument('--group_by', help='comma separated list of columns based on do the analysis', type=str,
	                    default="cluster_size,significance,min_subj,model,report,key")
	parser.add_argument('--model', help='Run analysis for a specific model', type=str)
	parser.add_argument('--min_folds', help='min number of folds for analysis and ranking. By default will be the '
	                                        'max number of folds', type=str)
	parser.add_argument('--output_dir', help='output_dir to save analysis output files. Default: ./analysis', type=str,
	                    default="analysis")

	args = parser.parse_args()

	execute(args)


def execute(args):
	if args.input_file is None or not (os.path.isfile(args.input_file)):
		print("bad input_file argument")
		exit(1)
	input_file = pd.read_csv(args.input_file)
	if args.model:
		input_file = input_file.loc[input_file[ 'model' ] == args.model, :]

	if args.group_by is None:
		print("missing or bad input_file argument")
		exit(1)

	group_by_columns = args.group_by.split(',')
	for column in group_by_columns:
		if column not in input_file.columns:
			print("{} is not in input_file columns".format(column))
			exit(1)

	df = None
	if args.processed_input_file is not None:
		if not (os.path.isfile(args.processed_input_file)):
			print("bad processed_input_file argument")
			exit(1)
		df = pd.read_csv(args.processed_input_file)
		if args.model:
			df = df.loc[df['model'] == args.model, :]

	if df is None:
		df = group_by_metric(input_file, group_by_columns)

	output_dir = args.output_dir
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)

	min_folds = args.min_folds
	if min_folds is None:
		min_folds = int(len(input_file['fold'].unique()) * 0.7)

	# filter only test reports
	df_filtered = df.loc[(df['report'] == 'test') & (df['n_folds'] >= min_folds) & (df['key'] == 'macro avg'), : ]

	df_filtered = df_filtered.sort_values(by='f1_score_mean', ascending=False);
	df_filtered.to_csv(os.path.join(output_dir, "accumlated_" + args.input_file), index=False)

	top_10 = df_filtered.head(10).loc[:, ['model', 'cluster_size', 'significance', 'min_subj', 'n_features_mean', 'f1_score_mean', 'n_folds']]
	top_10 = top_10.copy(deep=True)
	top_10['f1_score_mean'] = top_10['f1_score_mean'].apply(lambda x: round(x, 2))

	plt.clf()
	plt.figure()
	plt.table(cellText=top_10.to_numpy(), colLabels=top_10.columns, loc='center')
	plt.axis('off')
	plt.savefig(os.path.join(output_dir,  "top_models.png"), dpi=1200)

	plt.clf()
	sns.set(rc={'figure.figsize': (18, 10)})
	fig, axs = plt.subplots(ncols=2, nrows=2)

	best_model = df_filtered.iloc[0, df_filtered.columns == 'model'][0]
	print("best model is {} with f1_score_mean {}".format(best_model,
	                                                   df_filtered.iloc[0, df_filtered.columns == 'f1_score_mean'][0]))
	best_model_df = df_filtered.loc[df_filtered['model'] == best_model, :]
	best_model_df = best_model_df.sort_values(by='f1_score_mean', ascending=False);

	chart = sns.lineplot(data=best_model_df, x="n_features_mean", y="f1_score_mean", ax=axs[0, 0])
	ticks = np.arange(int(best_model_df['n_features_mean'].min()),
	                  int((best_model_df[ 'n_features_mean' ].max() - best_model_df['n_features_mean'].min()) / 10),
	                  int(best_model_df['n_features_mean'].max()))
	chart.set_xticks(ticks)
	chart.set_xticklabels(ticks, rotation=90)
	chart.set_title('Number of features vs f1 score')

	best_cluster_size = best_model_df.iloc[0, df_filtered.columns == 'cluster_size'][0]
	best_min_subj = best_model_df.iloc[0, df_filtered.columns == 'min_subj'][0]
	best_significance = best_model_df.iloc[0, df_filtered.columns == 'significance'][0]

	chart = sns.lineplot(data=best_model_df[(best_model_df["min_subj"] == best_min_subj) &
	                                        (best_model_df["significance"] == best_significance)],
	                     x="cluster_size",
	                     y="f1_score_mean",
	                     ax=axs[0, 1])
	chart.set_title('Cluster size vs f1 score')

	chart = sns.lineplot(data=best_model_df[(best_model_df["min_subj"] == best_min_subj) &
	                                        (best_model_df["cluster_size"] == best_cluster_size)],
	                     x="significance",
	                     y="f1_score_mean",
	                     ax=axs[1, 0])
	chart.set_title('Significance vs f1 score')

	chart = sns.lineplot(data=best_model_df[(best_model_df["cluster_size"] == best_cluster_size) &
	                                        (best_model_df["significance"] == best_significance)],
	                     x="min_subj",
	                     y="f1_score_mean",
	                     ax=axs[1, 1])
	chart.set_title('Min subjects vs f1 score')

	fig.savefig(os.path.join(output_dir, best_model + "_cluster_parameters_vs_f1_score.png"))

	if 'confusion_matrix' in best_model_df.columns:
		plt.clf()
		cm = pd.DataFrame(best_model_df.iloc[0, 'confusion_matrix'],
		                  index= best_model_df.iloc[0, 'labels'],
		                  columns= best_model_df.iloc[0, 'labels'])
		plt.figure(figsize=(10, 7))
		sns.heatmap(cm, annot=True)
		fig.savefig(os.path.join(output_dir, best_model + "_confusion_matrix.png"))

	plt.clf()
	indexing = (input_file['cluster_size'] == best_cluster_size) & (input_file['min_subj'] == best_min_subj) & \
	           (input_file['significance'] == best_significance) & (input_file['model'] == best_model) & \
	           (input_file['report'] == 'test') & (input_file['key'] == 'macro avg')
	chart = sns.distplot(a=input_file.loc[indexing, 'f1_score'], hist=True)
	chart.set_title(best_model + '_f1 score histogram')
	chart.get_figure().savefig(os.path.join(output_dir, best_model + '_f1_score_hist.png'))
	print(input_file.loc[indexing, 'parameters'].value_counts())

	plt.clf()
	best_models = df_filtered['model'].unique()[0:min(4, len(df_filtered['model'].unique()))]
	best_models_df_list = []
	for model in best_models:
		model_df = df_filtered.loc[df_filtered['model'] == model, : ]
		model_df = model_df.sort_values(by='f1_score_mean', ascending=False);
		model_cluster_size = model_df.iloc[0, df_filtered.columns == 'cluster_size'][0]
		model_min_subj= model_df.iloc[0, df_filtered.columns == 'min_subj'][0]
		model_significance= model_df.iloc[0, df_filtered.columns == 'significance'][0]
		indexing = (input_file['cluster_size'] == model_cluster_size) & (input_file['min_subj'] == model_min_subj) & \
		           (input_file['significance'] == model_significance) & (input_file['model'] == model) & \
		           (input_file['report'] == 'test') & (input_file['key'] == 'macro avg')
		best_models_df_list += [input_file.loc[ indexing, : ]]
	best_models_df = pd.concat(best_models_df_list)
	chart = sns.scatterplot(data=best_models_df, x="fold", y="f1_score", hue="model", x_jitter=0.2, y_jitter=0.2)
	chart.set_xticks(best_models_df['fold'].unique())
	chart.set_xticklabels(best_models_df['fold'].unique())
	chart.get_figure().savefig(os.path.join(output_dir, "f1_score_by_folds.png".format(len(best_models))))


if __name__ == "__main__":
    main()

