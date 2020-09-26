import os
import csv
import argparse
from collections import defaultdict
import math
import pandas as pd
import numpy as np
import time
import sys
from datetime import datetime
import logging
from sklearn.metrics.pairwise import pairwise_distances
import ray

ray.init(memory=20*1024*1024*1024, object_store_memory=20*1024*1024*1024)

logger = logging.getLogger(__name__)


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', "True"):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', "False"):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def read_vector_data(input_file):
    """Read vector data from input file and return it as pandas vector list"""
    logger.info(f"{datetime.now()} reading input_file {input_file}")
    from numpy import genfromtxt
    my_data = genfromtxt(input_file, delimiter=',', skip_header=1)
    return my_data


def write_output_file(output_file, proximity):
    header = ', '.join(str(el) for el in range(proximity.shape[1]))
    np.savetxt(output_file, proximity, delimiter=',', header=header)


def build_subject_status(data, id_field, label_field):
    """Returns a data frame where the for each id there's a single label

    e.g.:
    res[id_label][my_id] will be equal to the label assigned to patient my_id
    """
    subjects_db = data.loc[:, [id_field, label_field]]
    subjects_db.drop_duplicates(inplace=True)
    subjects_db.set_index(id_field, inplace=True)

    return subjects_db


def init_logger():
    logger.setLevel(logging.DEBUG)
    output_handler = logging.StreamHandler()
    output_handler.setLevel(logging.DEBUG)
    logs_dir = os.path.join('..','..','logs')
    if not(os.path.isdir(logs_dir)):
        os.mkdir(logs_dir)
    file_handler = logging.FileHandler(filename=os.path.join(logs_dir, 'log_'+ str(datetime.now().date()) + '.log'))
    output_handler.setLevel(logging.DEBUG)

    logger.addHandler(output_handler)
    logger.addHandler(file_handler)


def main():
    init_logger()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_file_path',
                        help='the filtered data file path')
    parser.add_argument('-v', '--vectors_file_path',
                        help='the vectors file path')
    parser.add_argument('-nn', '--NN_file_path',
                        help='the NN file path')
    parser.add_argument('-of', '--output_folder_path',
                        help='Output folder for the 3 output files - Nearest neighbors file, destances file, '
                             'and results anaylsis file')
    parser.add_argument('-od', '--output_description',
                        help='description to use inside output file names')
    parser.add_argument('--cpus', type=int, default=1, help='How many cores to run in parallel -  default is 1.')
    parser.add_argument('--steps', type=int, default=10000, help='How many rows to calculate in parallel, '
                                                                 'default is 10k.')
    parser.add_argument('--perform_NN', type=str2bool, default=True, help='Perform KD-tree and nearest neighbors '
                                                                          'analysis, and save NN list and distances')
    parser.add_argument('--perform_results_analysis', type=str2bool, default=False, help='Analyse nearest neighbors '
                                                                                         'file, to subject frequencies')
    args = parser.parse_args()
    if not(os.path.isfile(args.data_file_path)):
        print('feature file error, make sure feature and vectors file path\nExiting...')
        sys.exit(1)
        
    if args.perform_NN and not(os.path.isfile(args.vectors_file_path)):
        print('vectors file error, make sure feature and vectors file path\nExiting...')
        sys.exit(1)
        
    if not os.path.exists(args.output_folder_path):
        os.mkdir(args.output_folder_path)
        print('output_folder_created: ' + str(args.output_folder_path))

    if not(args.perform_NN) and not(os.path.isfile(args.NN_file_path)):
        print('Nearest neighbors file error, make sure the file exists\nExiting...')
        sys.exit(1)
    
    dateTimeObj = datetime.now()
    timeObj = dateTimeObj.time()
    logger.info(f'{timeObj} Beginning  data analysis')
    print('Data file path: ' + args.data_file_path)
    if args.perform_NN:
        print('Vectors file path: ' + args.vectors_file_path)
    else:
        print('NN file path: ' + args.NN_file_path)
        logger.info(f'NN file path: {args.NN_file_path}')
    print('-----------------------')
    output_file_name = args.output_description + '.csv'
    if args.perform_NN:
        cluster(args.vectors_file_path, args.output_folder_path, output_file_name, cluster_size=100, cpus=args.cpus,
                steps=args.steps)


@ray.remote
def build_distance_and_knn_maps(data, sub_row_range, k, cpus=2):
    t0 = time.time()
    distances_map = pairwise_distances(X=data[sub_row_range[0]:sub_row_range[1]], Y=data, metric='euclidean', n_jobs=cpus)
    print("build_distance_map: creating sub map took {}".format(time.time() - t0))

    t0 = time.time()
    knn_map = np.argpartition(distances_map, k, axis=1)[:, 0:k]
    print("build_knn_map: creating sub map took {}".format(time.time() - t0))

    return distances_map[:, 0:k], knn_map


def build_sub_map(data, major_row_range, cluster_size, cpus=2):
    step = (major_row_range[1]-major_row_range[0]) / cpus
    ranges = [[round(step * i), round(step * (i + 1))] for i in range(cpus)]
    results_ids = []

    knn_map = np.zeros(shape=[major_row_range[1]-major_row_range[0], cluster_size+1], dtype=int)
    distances_map = np.zeros(shape=[major_row_range[1]-major_row_range[0], cluster_size+1])
    for minor_row_range in ranges:
        sub_row_range = [major_row_range[0] + minor_row_range[0], major_row_range[0] + minor_row_range[1]]
        results_ids += [build_distance_and_knn_maps.remote(data, sub_row_range, k=cluster_size+1)]
    for i, minor_row_range in enumerate(ranges):
        sub_distances_map, sub_knn_map = ray.get(results_ids[i])
        distances_map[minor_row_range[0]:minor_row_range[1], :] = sub_distances_map
        knn_map[minor_row_range[0]:minor_row_range[1], :] = sub_knn_map

    return distances_map, knn_map


def build_maps(data, cluster_size, cpus=2, step=10000):
    knn_map = np.zeros(shape=[data.shape[0], cluster_size+1], dtype=int)
    distances_map = np.zeros(shape=[data.shape[0], cluster_size+1])
    ranges = [[round(step*i), round(step*(i+1))] for i in range(round(data.shape[0]/step))]

    for major_row_range in ranges:
        t0 = time.time()
        sub_distances_map, sub_knn_map = build_sub_map(data, major_row_range, cluster_size, cpus)
        print("build_maps: creating sub map (step {}) took {}".format(major_row_range[1], time.time()-t0))
        distances_map[major_row_range[0]:major_row_range[1], :] = sub_distances_map
        knn_map[major_row_range[0]:major_row_range[1], :] = sub_knn_map

    return distances_map, knn_map


def test_build_dist_and_knn_maps(dim_one=1000, dim_two=10, cluster_size=5, cpus=2):
    data = np.random.rand(dim_one, dim_two)
    t0 = time.time()
    distance_map, knn_map = build_maps(data, cluster_size, cpus=cpus, step=10)
    print("building maps completed after {}".format(time.time() - t0))


def cluster(data_file, output_file_path, output_file_name, cluster_size=100, cpus=2, steps=10000):
    """Cluster the data and write the result to output file"""
    vectors = read_vector_data(data_file)

    t0 = time.time()
    distance_map, knn_map = build_maps(vectors, cluster_size, cpus=cpus, steps=steps)
    print("cluster: building maps completed after {}".format(time.time() - t0))

    write_output_file(os.path.join(output_file_path, 'NN_' + output_file_name), knn_map)
    write_output_file(os.path.join(output_file_path, 'Distances_' + output_file_name),
                      distance_map.loc[:, 0:cluster_size])
    print(str(datetime.now()) + ' | finished clustering, files saved to: ')
    logger.info(str(datetime.now()) + ' | finished clustering, files saved to: ')
    logger.info(os.path.join(output_file_path, 'NN_' + output_file_name))
    logger.info(os.path.join(output_file_path, 'Distances_' + output_file_name))
    return knn_map


if __name__ == '__main__':
    main()

