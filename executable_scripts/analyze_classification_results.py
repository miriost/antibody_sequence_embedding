import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
	parser.add_argument('--processed_input_file', help='already processed classification results csv file')
	parser.add_argument('--group_by', help='comma separated list of columns based on do the analysis', type=str,
	                    default="cluster_size,significance,min_subj,model,report,key")
	parser.add_argument('--min_folds', help='min number of folds for analysis and ranking. By default will be the '
	                                        'max number of folds', type=str)
	parser.add_argument('--output_dir', help='output_dir to save analysis output files. Default: ./analysis', type=str,
	                    default="analysis")

	args = parser.parse_args()

	if args.input_file is None or not (os.path.isfile(args.input_file)):
		print("bad input_file argument")
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

	df = None
	if args.processed_input_file is not None:
		if not (os.path.isfile(args.processed_input_file)):
			print("bad processed_input_file argument")
			exit(1)
		df = pd.read_csv(args.processed_input_file)

	if df is None:
		df = group_by_metric(input_file, group_by_columns)

	output_dir = args.outputdir
	os.mkdir(output_dir)

	min_folds = parser.min_folds
	if min_folds is None:
		min_folds = int(len(input_file['fold'].unique()) * 0.7)

	# filter only test reports
	df_filtered = df.loc[(df['report'] == 'test') & (df['n_folds'] >= min_folds) & (df['key'] == 'macro avg'), : ]

	df_filtered = df_filtered.sort_values(by='f1_score_mean', ascending=False);
	df.to_csv(os.path.join(output_dir, "accumlated_" + args.input_file), index=False)

	best_model = df_filtered.iloc[0, df_filtered.columns == 'model'][0]
	print("best model is {} with f1_score_mean {}".format(best_model,
	                                                   df_filtered.iloc[0, df_filtered.columns == 'f1_score_mean'][0]))
	best_model_df = df_filtered.loc[df_filtered['model'] == best_model, :]
	best_model_df = best_model_df.sort_values(by='f1_score_mean', ascending=False);
	chart = sns.lineplot(data=best_model_df, x="n_features_mean", y="f1_score_mean")

	ticks = best_model_df['n_features_mean'].unique().astype(int).tolist()
	threshold = (max(ticks) - min(ticks)) / (len(ticks) * 2)
	max_score_tick = \
		int(best_model_df.loc[best_model_df['f1_score_mean'] == max(best_model_df['f1_score_mean']), "n_features_mean"].mean())
	min_score_tick = \
		int(best_model_df.loc[best_model_df['f1_score_mean'] == min(best_model_df['f1_score_mean']), "n_features_mean"].mean())
	ticks += [max_score_tick, min_score_tick]
	ticks.sort()

	ticks_labels = []
	for idx, tick in enumerate(ticks):
		if idx == 0 or tick in [max_score_tick, min_score_tick]:
			ticks_labels += [str(tick)]
		elif abs(tick-max_score_tick) > threshold and \
			abs(tick-min_score_tick) > threshold and \
			tick - ticks[idx-1] > threshold:
			ticks_labels += [str(tick)]
		else:
			ticks_labels += ['']

	chart.set_xticks(ticks)
	chart.set_xticklabels(ticks_labels, rotation=90)
	chart.get_figure().savefig(os.path.join(output_dir, "n_features_vs_f1_score.png"))

	best_cluster_size = best_model_df.iloc[0, df_filtered.columns == 'cluster_size'][0]
	best_min_subj= best_model_df.iloc[0, df_filtered.columns == 'min_subj'][0]
	best_significance= best_model_df.iloc[0, df_filtered.columns == 'significance'][0]

	plt.clf()
	chart = sns.lineplot(data=best_model_df[(best_model_df["min_subj"] == best_min_subj) &
	                                        (best_model_df["significance"] == best_significance)],
	                     x="cluster_size",
	                     y="f1_score_mean")
	chart.get_figure().savefig(os.path.join(output_dir, best_model + "_cluster_size_vs_f1_score.png"))
	chart = sns.lineplot(data=df_filtered, x="cluster_size", y="f1_score_mean")
	chart.get_figure().savefig(os.path.join(output_dir, "cluster_size_vs_f1_score.png"))

	plt.clf()
	chart = sns.lineplot(data=best_model_df[(best_model_df["min_subj"] == best_min_subj) &
	                                        (best_model_df["cluster_size"] == best_cluster_size)],
	                     x="significance",
	                     y="f1_score_mean")
	chart.get_figure().savefig(os.path.join(output_dir, best_model + "_significance_vs_f1_score.png"))
	chart = sns.lineplot(data=df_filtered, x="significance", y="f1_score_mean")
	chart.get_figure().savefig(os.path.join(output_dir, "significance_vs_f1_score.png"))

	plt.clf()
	chart = sns.lineplot(data=best_model_df[(best_model_df["cluster_size"] == best_cluster_size) &
	                                        (best_model_df["significance"] == best_significance)],
	                     x="min_subj",
	                     y="f1_score_mean")
	chart.get_figure().savefig(os.path.join(output_dir, best_model + "min_subj_vs_f1_score.png"))
	chart = sns.lineplot(data=df_filtered, x="min_subj", y="f1_score_mean")
	chart.get_figure().savefig(os.path.join(output_dir, "min_subj_vs_f1_score.png"))

	plt.clf()
	indexing = (input_file['cluster_size'] == best_cluster_size) & (input_file['min_subj'] == best_min_subj) & \
	           (input_file['significance'] == best_significance) & (input_file['model'] == best_model) & \
	           (input_file['report'] == 'test') & (input_file['key'] == 'macro avg')
	chart = sns.histplot(data=input_file.loc[indexing, :], x="f1_score")
	chart.get_figure().savefig(os.path.join(output_dir, "best_model_f1_score_hist.png"))

	plt.clf()
	best_models = df_filtered['model'].unique()[0:min(4, len(df_filtered['model'].unique()))]
	best_models_df_list = []
	for model in best_models:
		model_df = df_filtered.loc[ df_filtered[ 'model' ] == model, : ]
		model_df = model_df.sort_values(by='f1_score_mean', ascending=False);
		model_cluster_size = model_df.iloc[0, df_filtered.columns == 'cluster_size'][0]
		model_min_subj= model_df.iloc[0, df_filtered.columns == 'min_subj'][0]
		model_significance= model_df.iloc[0, df_filtered.columns == 'significance'][0]
		indexing = (input_file['cluster_size'] == model_cluster_size) & (input_file['min_subj'] == model_min_subj) & \
		           (input_file['significance'] == model_significance) & (input_file['model'] == model) & \
		           (input_file['report'] == 'test') & (input_file['key'] == 'macro avg')
		best_models_df_list += [input_file.loc[ indexing, : ]]
	best_models_df = pd.concat(best_models_df_list)
	chart = sns.scatterplot(data=best_models_df, x="fold", y="f1_score", hue="model")
	chart.set_xticks(best_models_df['fold'].unique())
	chart.set_xticklabels(best_models_df['fold'].unique())
	chart.get_figure().savefig(os.path.join(output_dir, "{}_best_models.png".format(len(best_models))))

