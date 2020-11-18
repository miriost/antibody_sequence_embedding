import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import ast


def group_by_metric(df: pd.DataFrame, metric_list: list) -> pd.DataFrame:

    if len(metric_list) == 0:
        desc = df.loc[:, ['accuracy', 'f1_score', 'precision', 'recall']].describe()
        if 'confusion_matrix' in df.columns:
            res = pd.DataFrame(index=[0], columns=['confusion_matrix'])
        else:
            res = pd.DataFrame(index=[0])
        for idx, row in desc.iterrows():
            res[row.add_suffix('_' + idx).index.values.tolist()] = pd.DataFrame([row.values.tolist()], index=res.index)
        res['n_features_mean'] = round(df['n_features'].mean())
        res['n_folds'] = len(df['fold'].unique())
        if 'labels' in df.columns:
            res['labels'] = df.iloc[0, np.where(df.columns == 'labels')[0][0]]
        res['parameters'] = format(df['parameters'].value_counts().to_dict())
        if 'confusion_matrix' in df.columns:
            arr = df['confusion_matrix'].apply(lambda x: np.array(json.loads(x)))
            res.at[0, 'confusion_matrix'] = sum(arr.to_list()).tolist()

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
                        default="cluster_size,max_features,model,report,key")
    parser.add_argument('--model', help='Run analysis for a specific model', type=str)
    parser.add_argument('--numeric_hyper_params', help='Semicolon separated list of Numeric generic hyper parameters '
                                                       'in csv file', type=str)
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

    if args.numeric_hyper_params is not None:
        numeric_hyper_params = args.numeric_hyper_params.split(';')
    else:
        numeric_hyper_params = []

    # filter only test reports
    df_filtered = df.loc[(df['report'] == 'test') & (df['n_folds'] >= min_folds) & (df['key'] == 'weighted avg'), : ]

    df_filtered = df_filtered.sort_values(by='f1_score_mean', ascending=False);
    df_filtered.to_csv(os.path.join(output_dir, "accumulated_" + args.input_file), index=False)

    models_table_columns = ['model', 'n_folds', 'cluster_size', 'max_features', 'n_features_mean', 'f1_score_mean']

    models_list = [df_filtered[df_filtered['model'] == model].iloc[0, :].loc[models_table_columns] for
                   model in df_filtered['model'].unique()]
    models_df = pd.concat(models_list, axis=1).transpose()
    models_df['f1_score_mean'] = models_df['f1_score_mean'].apply(lambda x: round(x, 2))
    print(models_df)

    plt.clf()
    plt.figure()
    plt.table(cellText=models_df.to_numpy(), colLabels=models_df.columns, loc='center')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir,  "top_models.png"), dpi=1200)
    plt.clf()

    best_model = df_filtered.iloc[0, df_filtered.columns == 'model'][0]
    best_model_df = df_filtered.loc[df_filtered['model'] == best_model, :]
    best_model_df = best_model_df.sort_values(by='f1_score_mean', ascending=False);
    best_model_df.reset_index(inplace=True)

    chart = sns.lineplot(data=best_model_df, x="n_features_mean", y="f1_score_mean")
    ticks = np.arange(int(best_model_df['n_features_mean'].min()),
                      int((best_model_df[ 'n_features_mean'].max() - best_model_df['n_features_mean'].min()) / 10),
                      int(best_model_df['n_features_mean'].max()))
    chart.set_xticks(ticks)
    chart.set_xticklabels(ticks, rotation=90)
    chart.set_title('Number of features vs f1 score')

    chart.get_figure().savefig(os.path.join(output_dir, best_model + "_cluster_parameters_vs_f1_score.png"))
    plt.clf()

    if 'confusion_matrix' in best_model_df.columns:
        cm_arr = best_model_df.loc[0, 'confusion_matrix']
        labels_str = best_model_df.loc[0, 'labels']
        labels = ast.literal_eval(labels_str)
        cm = pd.DataFrame(cm_arr,
                          index=labels,
                          columns=labels)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True)
        plt.savefig(os.path.join(output_dir, best_model + "_confusion_matrix.png"))
        plt.clf()

    df = input_file[(input_file['model'] == best_model) & (input_file['report'] == 'test') & (input_file['key'] == 'macro avg')]
    for param in numeric_hyper_params:
        df = df[param == best_model_df.at[best_model_df.index[0], param]]

    chart = sns.distplot(a=df['f1_score'], hist=True)
    chart.set_title(best_model + '_f1 score histogram')
    chart.get_figure().savefig(os.path.join(output_dir, best_model + '_f1_score_hist.png'))
    plt.clf()


if __name__ == "__main__":
    main()


