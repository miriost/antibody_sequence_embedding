import pandas as pd
import argparse
import os, sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json


def main():
    parser = argparse.ArgumentParser(description='create analysis and plots based on the clustering files')

    parser.add_argument('--analysis_file', help='input cluster analysis csv file')
    parser.add_argument('--labels', help='semicolon separated list of clustering target labels')
    parser.add_argument('--output_dir', help='semicolon separated list of clustering target labels', default='.')
    parser.add_argument('--features_file', help='feature list file to mark selected features in plot')

    args = parser.parse_args()
    execute(args)


def execute(args):

    if args.labels is None:
        print('Missing mandatory labels argument. Exiting...')
        sys.exit(1)
    labels = args.labels.split(';')

    if args.analysis_file is None or not os.path.isfile(args.analysis_file):
        print('Bad or missing analysis_file argument. Existing...')
        sys.exit(1)

    analysis_file = pd.read_csv(args.analysis_file, sep='\t')
    for label in labels:
        if label not in analysis_file.columns:
            print("{} is not in data file columns: {}\nExisting...".format(label, analysis_file.columns))
            sys.exit(1)

    colors = np.array(["b"] * len(analysis_file))
    if args.features_file:
        feature_file = pd.read_csv(args.features_file)
        features = analysis_file['vector'].apply(lambda x: x in feature_file['vector'].tolist())
        colors[features] = "r"

    for label in labels:
        g = sns.jointplot(x='how_many_subjects', y=label, data=analysis_file, kind="kde",
                          color="m")
        g.plot_joint(plt.scatter, s=30, linewidth=1, marker='+', color=colors.tolist())
        g.ax_joint.collections[0].set_alpha(0)
        g.set_axis_labels("Number of subjects in cluster", "% of " + label + " in cluster")
        g.savefig(os.path.join(args.output_dir, label + '_distribution_in_cluster.png'))
        plt.clf()


if __name__ == '__main__':
    main()
