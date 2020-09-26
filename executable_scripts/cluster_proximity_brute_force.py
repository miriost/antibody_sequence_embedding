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


def write_output_file(output_file, proximity, fmt=None):
    header = ', '.join(str(el) for el in range(proximity.shape[1]))
    if fmt is None:
        np.savetxt(output_file, proximity, delimiter=',', header=header)
    else:
        np.savetxt(output_file, proximity, delimiter=',', header=header, fmt=fmt)


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


def build_subject_status(data, id_field, label_field):
    """Returns a data frame where the for each id there's a single label
    e.g.:
    res[id_label][my_id] will be equal to the label assigned to patient my_id
    """
    subjects_db = data.loc[:, [id_field, label_field]]
    subjects_db.drop_duplicates(inplace=True)
    subjects_db.set_index(id_field, inplace=True)

    return subjects_db


def get_subject_stats(subjects, subject_status, status_types):
    """Return a dataframe where for each status theres the percentage of subjects


    status1 | status2 | status3
    60%     | 25%     | 15%

    @:argument subjects: the list of subjects
    @:argument subjects_status: a dict where for every subject there's his status
    @:return a data frame 3x1
    """
    subset = subject_status.loc[subjects, subject_status.columns[0]]
    stats = subset.value_counts()

    d = {status: (1.0*stats.get(status, 0.0)/len(subset)) for status in status_types}
    return d


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
    parser.add_argument('--step', type=int, default=10000, help='How many rows to calculate in parallel, '
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
        knn_map = cluster(args.vectors_file_path, args.output_folder_path, output_file_name, cluster_size=100,
                          cpus=args.cpus, step=args.step)

    if args.perform_results_analysis:
        if not args.perform_NN:
            knn_map = np.loadtxt(args.NN_file_path, delimiter=',', skiprows=1)
        out = analyze_data(knn_map, args.data_file_path, cpus=args.cpus)
        output_file = args.output_description + '_analysis.csv'
        out.to_csv(os.path.join(args.output_folder_path, output_file))
        logger.info(str(datetime.now()) + ' | data written to output file: ' + output_file)


def analyze_data(knn_map, data_file, id_field='SUBJECT', status_field='labels'):
    logger.info(f'{str(datetime.now())} | Begin data analyze of nearest neighbors')
    data = pd.read_csv(data_file, sep='\t')
    status_types = data[status_field].unique()
    subject_status = build_subject_status(data, id_field, status_field)
    logger.info(f'{str(datetime.now())} | Build subject status complete')
    list_out = [np.nan]*len(knn_map)
    fields_to_extract = [id_field, status_field]
    t0 = time.time()
    for idx, row in enumerate(knn_map):

        cluster_data = data.loc[row, fields_to_extract]
        subjects = cluster_data[id_field].unique()
        stats = get_subject_stats(subjects, subject_status, status_types)
        res = {
            'neighbors': [int(x) for x in row],
            'how_many_subjects': len(subjects),
        }
        res.update(stats)
        list_out[idx] = res

        if idx % 1000 == 0:
            logger.info('{} | index = {}, finished 1000 vectors in {:.3} sec'.format(str(datetime.now()), str(row[:6]), time.time()-t0))
            t0 = time.time()

    output_df = pd.DataFrame(list_out,columns = ['neighbors', 'how_many_subjects'].extend(status_types))
    logger.info(f'{str(datetime.now())} | Finished analysis and transfered to dataframe')
    return output_df


@ray.remote
def build_distance_and_knn_maps(data, sub_row_range, k, cpus=2):
    t0 = time.time()
    print("building distance map for range {}".format(sub_row_range))
    distances_map = pairwise_distances(X=data[sub_row_range[0]:sub_row_range[1]], Y=data, metric='euclidean',
                                       n_jobs=cpus)
    distances_map = np.sort(distances_map, axis=1)
    print("building distance map for range {} took {}".format(sub_row_range, time.time() - t0))

    t0 = time.time()
    print("building knn map for range {}".format(sub_row_range))
    knn_map = np.argpartition(distances_map, k, axis=1)[:, 0:k]
    print("building knn map for range {} took {}".format(sub_row_range, time.time() - t0))

    return distances_map[np.arange(distances_map.shape[0])[:, None], knn_map], knn_map


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
    distances_map = np.zeros(shape=[data.shape[0], cluster_size+1])
    knn_map = np.zeros(shape=[data.shape[0], cluster_size+1], dtype=int)
    ranges = [[round(step*i), round(step*(i+1))] for i in range(round(data.shape[0]/step))]

    for major_row_range in ranges:
        t0 = time.time()
        print("calling build_sub_map for range {}".format(major_row_range))
        sub_distances_map, sub_knn_map = build_sub_map(data, major_row_range, cluster_size, cpus)
        print("build_sub_map: creating sub map (range {}) took {}".format(major_row_range, time.time()-t0))
        distances_map[major_row_range[0]:major_row_range[1], :] = sub_distances_map
        knn_map[major_row_range[0]:major_row_range[1], :] = sub_knn_map

    return distances_map, knn_map


def test_build_dist_and_knn_maps(dim_one=1000, dim_two=10, cluster_size=5, cpus=2):
    data = np.random.rand(dim_one, dim_two)
    t0 = time.time()
    distance_map, knn_map = build_maps(data, cluster_size, cpus=cpus, step=10)
    print("building maps completed after {}".format(time.time() - t0))


def cluster(data_file, output_file_path, output_file_name, cluster_size=100, cpus=2, step=10000):
    """Cluster the data and write the result to output file"""
    vectors = read_vector_data(data_file)

    t0 = time.time()
    distance_map, knn_map = build_maps(vectors, cluster_size, cpus=cpus, step=step)
    print("cluster: building maps completed after {}".format(time.time() - t0))

    write_output_file(os.path.join(output_file_path, 'NN_' + output_file_name), knn_map, fmt="%d")
    write_output_file(os.path.join(output_file_path, 'Distances_' + output_file_name), distance_map)
    print(str(datetime.now()) + ' | finished clustering, files saved to: ')
    logger.info(str(datetime.now()) + ' | finished clustering, files saved to: ')
    logger.info(os.path.join(output_file_path, 'NN_' + output_file_name))
    logger.info(os.path.join(output_file_path, 'Distances_' + output_file_name))
    return knn_map


if __name__ == '__main__':
    main()

