import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os
import sys
import math


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', help='path to the working directory, default is current directory', default='./')
    parser.add_argument('--distance_metrics', help="semicolon separated list of prefixes of the distance files for "
                                                   "comparision, default is 'euclidean'", default='euclidean')
    parser.add_argument('--same_cdr3_length', help='search knn among sequences with the same cdr3 length, default is '
                                                   'False', default=False, type=str2bool)
    args = parser.parse_args()

    execute(args)


def execute(args):

    work_dir = args.work_dir
    dist_metrics = args.distance_metrics.split(';')
    same_cdr3_length = args.same_cdr3_length

    if same_cdr3_length is True:
        prefix = 'same_cdr3_length_'
    else:
        prefix = ''

    lev_dist_file = os.path.join(work_dir, prefix + 'levenshtein_distances_map.npy')
    if not os.path.isfile(lev_dist_file):
        print('Cannot find file {}\nExiting...'.format(lev_dist_file))
        sys.exit(1)
    lev_dist_map = np.load(lev_dist_file)

    plt.clf()
    fig_dim = math.ceil(len(dist_metrics)/2)
    fig, axes = plt.subplots(ncols=fig_dim, nrows=fig_dim, constrained_layout=True)

    if fig_dim == 1:
        axes = np.array([axes]).reshape([1, -1])

    if same_cdr3_length is False:
        fig.suptitle('Immune2vec vector space vs Levenshtein distance')
    else:
        fig.suptitle('Immune2vec vector space vs Levenshtein distance (same CDR3 length)')

    colors = sns.color_palette()
    for idx, dist_metric in enumerate(dist_metrics):
        if idx == axes.size:
            break
        metric_dist_file = os.path.join(work_dir, prefix + dist_metric + '_distances_map.npy')
        if not os.path.isfile(metric_dist_file):
            print('Cannot find file {}\nSkipping...'.format(metric_dist_file))
        metric_dist_map = np.load(metric_dist_file)
        ax = axes[int(idx / axes.shape[1]), idx % axes.shape[0]]
        sns.scatterplot(x=lev_dist_map.flatten(), y=metric_dist_map.flatten(), ax=ax, color=colors[idx])
        dots = ax.collections[-1]
        offsets = dots.get_offsets()
        jitter_offsets = offsets + np.random.uniform(-0.005, 0.005, offsets.shape)
        dots.set_offsets(jitter_offsets)
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.set_title(dist_metric)

    print('saving file ' + os.path.join(work_dir, prefix + 'distance_comparing.png'))
    fig.savefig(os.path.join(work_dir, prefix + 'distance_comparing.png'))


if __name__ == '__main__':
    main()
