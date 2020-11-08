import os
import argparse
import math
import pandas as pd
import numpy as np
import time
import sys
from datetime import datetime
import logging
from sklearn.metrics.pairwise import pairwise_distances
import ray
import json
import random
import ast

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

def write_vector_file(output_file, proximity, fmt=None):
    header = ', '.join(str(el) for el in range(proximity.shape[1]))
    if fmt is None:
        np.savetxt(output_file, proximity, delimiter=',', header=header)
    else:
        np.savetxt(output_file, proximity, delimiter=',', header=header, fmt=fmt)


def build_subject_status(data, id_column, label_field):
    """Returns a data frame where the for each id there's a single label

    e.g.:
    res[id_label][my_id] will be equal to the label assigned to patient my_id
    """
    subjects_db = data.loc[:, [id_column, label_field]]
    subjects_db.drop_duplicates(inplace=True)
    subjects_db.set_index(id_column, inplace=True)

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


def build_subject_status(data, id_column, label_field):
    """Returns a data frame where the for each id there's a single label
    e.g.:
    res[id_label][my_id] will be equal to the label assigned to patient my_id
    """
    subjects_db = data.loc[:, [id_column, label_field]]
    subjects_db.drop_duplicates(inplace=True)
    subjects_db.set_index(id_column, inplace=True)

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
    parser.add_argument('--data_file_path',
                        help='the filtered data file path')
    parser.add_argument('--vector_column',
                        help='the name of the column with the vector')
    parser.add_argument('--label_column', default='repertoire.disease_diagnosis',
                        help='the name of the label column')
    parser.add_argument('--NN_file_path',
                        help='the NN file path')
    parser.add_argument('--output_folder_path',
                        help='Output folder for the 3 output files - Nearest neighbors file, destances file, '
                             'and results anaylsis file')
    parser.add_argument('--output_description',
                        help='description to use inside output file names')
    parser.add_argument('--cpus', type=int, default=1, help='How many cores to run in parallel -  default is 1.')
    parser.add_argument('--step', type=int, default=10000, help='How many rows to calculate in parallel, '
                                                                 'default is 10k.')
    parser.add_argument('--perform_NN', type=str2bool, default=True, help='Perform KD-tree and nearest neighbors '
                                                                          'analysis, and save NN list and distances')
    parser.add_argument('--perform_results_analysis', type=str2bool, default=False, help='Analyse nearest neighbors '
                                                                                         'file, to subject frequencies')
    parser.add_argument('--dist_metric',
                        help='type of distance to use, default=euclidean', default='euclidean', type=str)
    parser.add_argument('--thread_memory', help='memory size for ray thread (bytes)', type=int)
    parser.add_argument('--cluster_size', help='size of the cluster, deafult=100', type=int, default=100)
    parser.add_argument('--id_column', help='name of the subject id column, default=repertoire.repertoire_name',
                        type=str, default='repertoire.repertoire_name')

    args = parser.parse_args()
    if not os.path.isfile(args.data_file_path):
        print('Data file error, make sure data file path: {}\nExiting...'.format(args.data_file_path))
        sys.exit(1)
        
    if not os.path.exists(args.output_folder_path):
        os.mkdir(args.output_folder_path)
        print('output_folder_created: ' + str(args.output_folder_path))

    if not(args.perform_NN) and not(os.path.isfile(args.NN_file_path)):
        print('Nearest neighbors file error, make sure the file exists: path {}\nExiting...'.format(args.NN_file_path))
        sys.exit(1)

    if args.vector_column is None:
        print("Missing vector_column argument\nExisting...")
        sys.exit(1)

    if args.thread_memory is not None:
        ray.init(memory=args.thread_memory, object_store_memory=args.thread_memory)
    else:
        ray.init()
   

    date_time_obj = datetime.now()
    time_obj = date_time_obj.time()

    logger.info(f'{time_obj} Beginning  data analysis')
    print('Data file path: ' + args.data_file_path)
    if not args.perform_NN:
        print('NN file path: ' + args.NN_file_path)
        logger.info(f'NN file path: {args.NN_file_path}')
    print('-----------------------')
    output_file_name = args.output_description

    data_file = pd.read_csv(args.data_file_path, sep='\t')
    grouped_by_vector = data_file.groupby(by=args.vector_column)
    points = grouped_by_vector[args.id_column].apply(np.unique).reset_index(name='vector_subjects')
    points['vector_subjects'] = list(map(lambda x: x.tolist(), points['vector_subjects']))
    points['vector_duplicate_count'] = grouped_by_vector[args.id_column].apply(len).reset_index(name='len')['len']

    if args.perform_NN:
        vectors = np.array(points[args.vector_column].apply(lambda x: json.loads(x)).to_list())
        knn_map, distance_map = cluster(vectors, args.output_folder_path, output_file_name,
                                        cluster_size=args.cluster_size, dist_metric=args.dist_metric, cpus=args.cpus,
                                        step=args.step)
        knn_info = pd.DataFrame(knn_map, columns=range(knn_map.shape[1]))
        knn_info['median_distance'] = np.median(distance_map, axis=1)
        knn_info['max_distance'] = np.max(distance_map, axis=1)
        knn_info['vector_subjects'] = points['vector_subjects']
        knn_info['vector'] = points[args.vector_column]
        knn_info['vector_duplicate_count'] = points['vector_duplicate_count']
        knn_info.to_csv(os.path.join(args.output_folder_path, 'NN_' + output_file_name + '.tsv'), sep='\t', index=False)
        knn_info['vector_subjects'] = points['vector_subjects']
        logger.info(os.path.join(args.output_folder_path, 'NN_' + output_file_name + '.tsv'))

    if args.perform_results_analysis:
        if not args.perform_NN:
            knn_info = pd.read_csv(args.NN_file_path, dtype={i: 'int' for i in range(args.cluster_size + 1)}, sep='\t')
            knn_info['vector_subjects'] = knn_info['vector_subjects'].apply(lambda x: ast.literal_eval(x))
            knn_map = knn_info.loc[:, [str(x) for x in range(args.cluster_size + 1)]].to_numpy()
        out = analyze_data(knn_info, knn_map, data_file, id_column=args.id_column, label_column=args.label_column,
                           cpus=args.cpus)
        out['median_distance'] = knn_info['median_distance'].astype(float)
        out['max_distance'] = knn_info['max_distance'].astype(float)
        out['vector'] = knn_info['vector'].astype(float)
        output_file = args.output_description + '_analysis.csv'
        out.to_csv(os.path.join(args.output_folder_path, output_file), index=False)
        logger.info(str(datetime.now()) + ' | data written to output file: ' + output_file)


def analyze_data(knn_info: pd.DataFrame, knn_map: np.ndarray, data_file: pd.DataFrame,
                 id_column='repertoire.repertoire_name', label_column='repertoire.disease_diagnosis', cpus=2):

    logger.info(f'{str(datetime.now())} | Begin data analyze of nearest neighbors')
    status_types = data_file[label_column].unique().tolist()

    step = len(knn_map)/cpus
    ranges = [[round(step * i), round(step * (i + 1))] for i in range(cpus)]

    subjects = data_file.loc[:, ].groupby(by=[id_column])[label_column].apply(lambda x: x.iloc[0])

    results_ids = []
    for sub_range in ranges:
        results_ids += [analyze_sub_data.remote(knn_info, knn_map, sub_range, status_types, subjects)]

    output_df = pd.concat([ray.get(result_id) for result_id in results_ids], ignore_index=True)

    return output_df


@ray.remote
def analyze_sub_data(knn_info: pd.DataFrame, knn_map: np.ndarray, sub_range: tuple, status_types: list,
                     subjects: pd.DataFrame):
    print("adding neighbors column: range {}".format(sub_range))
    t0 = time.time()
    sub_output_df = pd.DataFrame(columns=['neighbors', 'how_many_subjects'] + status_types)
    sub_output_df['neighbors'] = knn_map[sub_range[0]:sub_range[1], :].tolist()
    print("neighbors column added, took {}".format(time.time() - t0))

    print("adding how_many_subjects column: range {}".format(sub_range))
    t0 = time.time()
    
    print(len(knn_map))
    print(len(knn_info))
    neighbors = np.array(knn_info[['vector_subjects']].transpose())[np.arange(1)[:, None], knn_map[sub_range[0]:sub_range[1], :]]
    neighbors = np.array(list(map(lambda x: np.unique(np.concatenate(x)), neighbors)))

    sub_output_df['subjects'] = list(map(lambda x: x.tolist(), neighbors))
    sub_output_df['how_many_subjects'] = list(map(lambda x: len(x), neighbors))
    print("how_many_subjects column added, took {}".format(time.time() - t0))

    print("adding status columns: range {}".format(sub_range))
    t0 = time.time()
    tmp = pd.concat(list(map(lambda x: subjects[x].value_counts(normalize=True), neighbors)), sort=True, axis=1)
    tmp = tmp.transpose().reset_index(drop=True)
    tmp.fillna(0)
    sub_output_df[status_types] = tmp[status_types]

    print("status columns added, took {}".format(time.time() - t0))

    return sub_output_df


def test_analyze_sub_data():
    data = pd.DataFrame()
    data['subject'] = random.choices(['P1_I1', 'P1_I2', 'P1_I3', 'P1_I4', 'P1_I5', 'P1_I6', 'P1_I7', 'P1_I8'], k=1000)
    data['labels'] = random.choices(['healthy', 'celiac'], k=1000)
    knn_map = np.array([random.sample(range(1000), 10) for i in range(1000)])
    status_types = data['labels'].unique()

    cpus = 3
    step = len(data)/cpus
    ranges = [[round(step * i), round(step * (i + 1))] for i in range(cpus)]

    results_ids = []
    for sub_range in ranges:
        results_ids += [analyze_sub_data.remote(knn_map, data, sub_range, status_types)]

    output_df = pd.concat([ray.get(result_id) for result_id in results_ids], ignore_index=True)

    return output_df


@ray.remote
def build_distance_and_knn_maps(data, sub_row_range, k, dist_metric='euclidean', cpus=2):
    t0 = time.time()
    print("building distance map for range {}".format(sub_row_range))
    distances_map = pairwise_distances(X=data[sub_row_range[0]:sub_row_range[1]], Y=data, metric=dist_metric,
                                       n_jobs=cpus)
    print("building distance map for range {} took {}".format(sub_row_range, time.time() - t0))

    t0 = time.time()
    print("building knn map for range {}".format(sub_row_range))
    knn_map = np.argpartition(distances_map, k, axis=1)[:, 0:k]
    knn_map = knn_map[np.arange(knn_map.shape[0])[:, None],
                      np.argsort(distances_map[np.arange(distances_map.shape[0])[:, None], knn_map])]
    print("building knn map for range {} took {}".format(sub_row_range, time.time() - t0))

    return distances_map[np.arange(distances_map.shape[0])[:, None], knn_map], knn_map


def build_sub_map(data, major_row_range, cluster_size, dist_metric='euclidean', cpus=2):
    step = (major_row_range[1]-major_row_range[0]) / cpus
    ranges = [[round(step * i), round(step * (i + 1))] for i in range(cpus)]
    results_ids = []

    knn_map = np.zeros(shape=[major_row_range[1]-major_row_range[0], cluster_size+1], dtype=int)
    distances_map = np.zeros(shape=[major_row_range[1]-major_row_range[0], cluster_size+1])
    for minor_row_range in ranges:
        sub_row_range = [major_row_range[0] + minor_row_range[0], major_row_range[0] + minor_row_range[1]]
        results_ids += [build_distance_and_knn_maps.remote(data, sub_row_range, k=cluster_size+1,
                                                           dist_metric=dist_metric, cpus=cpus)]
    for i, minor_row_range in enumerate(ranges):
        sub_distances_map, sub_knn_map = ray.get(results_ids[i])
        distances_map[minor_row_range[0]:minor_row_range[1], :] = sub_distances_map
        knn_map[minor_row_range[0]:minor_row_range[1], :] = sub_knn_map

    return distances_map, knn_map


def build_maps(data, cluster_size, dist_metric='euclidean', cpus=2, step=15000):
    distances_map = np.zeros(shape=[data.shape[0], cluster_size+1])
    knn_map = np.zeros(shape=[data.shape[0], cluster_size+1], dtype=int)
    partitions = math.ceil(data.shape[0] / step)
    ranges = [[round(step*i), min(round(step*(i+1)), data.shape[0])] for i in range(partitions)]

    for major_row_range in ranges:
        t0 = time.time()
        print("calling build_sub_map for range {}".format(major_row_range))
        sub_distances_map, sub_knn_map = build_sub_map(data, major_row_range, cluster_size=cluster_size,
                                                       dist_metric=dist_metric, cpus=cpus)
        print("build_sub_map: creating sub map (range {}) took {}".format(major_row_range, time.time()-t0))
        distances_map[major_row_range[0]:major_row_range[1], :] = sub_distances_map
        knn_map[major_row_range[0]:major_row_range[1], :] = sub_knn_map

    return distances_map, knn_map


def test_build_dist_and_knn_maps(dim_one=1000, dim_two=10, cluster_size=5, cpus=2):
    data = np.random.rand(dim_one, dim_two)
    t0 = time.time()
    distance_map, knn_map = build_maps(data, cluster_size, dist_metric='euclidean', cpus=cpus, step=10)
    print("building maps completed after {}".format(time.time() - t0))


def cluster(vectors, output_file_path, output_file_name, cluster_size=100, dist_metric='euclidean',
            cpus=2, step=10000):
    """Cluster the data and write the result to output file"""
    t0 = time.time()
    distance_map, knn_map = build_maps(vectors, cluster_size, dist_metric=dist_metric, cpus=cpus, step=step)
    print("cluster: building maps completed after {}".format(time.time() - t0))

    write_vector_file(os.path.join(output_file_path, 'Distances_' + output_file_name + '.csv'), distance_map)
    print(str(datetime.now()) + ' | finished clustering, files saved to: ')
    logger.info(str(datetime.now()) + ' | finished clustering, files saved to: ')
    logger.info(os.path.join(output_file_path, 'Distances_' + output_file_name + '.csv'))

    return knn_map, distance_map


if __name__ == '__main__':
    main()




