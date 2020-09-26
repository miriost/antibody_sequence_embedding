import os
import csv
import argparse
from collections import defaultdict
import math

import pandas as pd
import numpy as np
from scipy.spatial import cKDTree, KDTree
import time
import sys
import multiprocessing
#import pprofile
#import cProfile
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


#  convert a string to a its boolean representation
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', "True"):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', "False"):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def date_time_obj():
    init_logger()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_file_path', help='the tab filtered data file path')
    parser.add_argument('-v', '--vectors_file_path', help='the csv vectors file path')
    parser.add_argument('-nn', '--NN_file_path', help='the NN file path')
    parser.add_argument('-of', '--output_folder_path',  help='Output folder for the 3 output files - Nearest neighbors '
                                                             'file, distances file, and results analysis file')
    parser.add_argument('-od', '--output_description',  help='description to use inside output file names')
    parser.add_argument('--cpus', type=int, default=1, help='How many cores to run in parallel')
    parser.add_argument('--perform_NN', type=str2bool, default=True, help='Perform KD-tree and nearest neighbors'
                                                                          ' analysis, and save NN list and distances')
    parser.add_argument('--perform_results_analysis', type=str2bool, default=False, help='Analyse nearest neighbors '
                                                                                         'file, to subject frequencies')
    parser.add_argument('-l', '--label_column', type=str, default='labels', help='name of the labels column, '
                                                                                 'default is "labels"')

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
        neighbors_list = cluster(args.vectors_file_path, args.output_folder_path,
                                 output_file_name, cluster_size=100, cpus=args.cpus)

    if args.perform_results_analysis:
        if not(args.perform_NN): #need to load neigbors list as it was not processed
            neighbors_list = np.loadtxt(args.NN_file_path, delimiter=',', skiprows =1)

        if args.cpus == 1:
            out = analyze_data(neighbors_list, args.data_file_path)
        else:
            out = analyze_data_parallel(neighbors_list, args.data_file_path, cpus=args.cpus)
        output_file = args.output_description + '_analysis.csv'
        out.to_csv(os.path.join(args.output_folder_path, output_file))
        logger.info(str(datetime.now()) +  ' | data written to output file: ' + output_file)


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


def read_vector_data(input_file):
    """Read vector data from input file and return it as pandas vector list"""
    logger.info(f"{datetime.now()} reading input_file {input_file}")
    from numpy import genfromtxt
    my_data = genfromtxt(input_file, delimiter=',', skip_header=1)
    return my_data


def write_proximity_file(output_file, proximity):
    header = ', '.join(str(el) for el in range(proximity.shape[1]))
    np.savetxt(output_file, proximity, delimiter=',', header=header)


def get_proximity_list(data, cluster_size):
    proximity_list = np.zeros([data.shape[0], cluster_size+1])
    distances_list = np.zeros([data.shape[0], cluster_size+1])
        # Build KDTree to inquire the closest points
    tree = KDTree(data)
    print(str(datetime.now()) + '| KDtree ready')
    t0 = time.time()
    upper_bound = 10
    for vector_idx, vector in enumerate(data):      
        distances, indices = tree.query(vector,cluster_size+1,p=2,distance_upper_bound=upper_bound)
        while math.inf in distances:
            upper_bound = upper_bound*1.1
            distances, indices = tree.query(vector, cluster_size + 1,p=2,distance_upper_bound=upper_bound)
            print('Upper bound updated itertivly to: ' + str(upper_bound))
        proximity_list[vector_idx] = np.array(indices, dtype=np.int)
#        print(proximity_list[vector_idx])
        distances_list[vector_idx] = np.array(distances, dtype=np.float)
#        print(distances_list[vector_idx])
        if vector_idx%1000==0 and vector_idx>0:
            dateTimeObj = datetime.now()
            timeObj = dateTimeObj.time()
            print('{} | finished 1000 vectors in {:.3} sec'.format(timeObj, time.time()-t0))
            distances_max = np.max(distances_list[(vector_idx-1000):(vector_idx+1)])
            print('Maximal distance between neighbors, indexes {} to {}: {}'.format((vector_idx-1000),(vector_idx+1), distances_max))
            upper_bound = (upper_bound*(vector_idx//1000) + distances_max)/(1+vector_idx//1000)
            print('Updating upper bound to:' + str(upper_bound))
            t0 = time.time()

    return proximity_list, distances_list


def worker_query(proximity_list, distances_list, vector_idx, tree, vector, 
                 cluster_size, p, upper_bound, out_q):
    """Run a single query and return the result"""
    # import ipdb; ipdb.set_trace()
    distances, indices = tree.query(vector,cluster_size,p=2,distance_upper_bound=upper_bound)
    while math.inf in distances:
            upper_bound = upper_bound*1.1
            distances, indices = tree.query(vector, cluster_size,p=2,distance_upper_bound=upper_bound)
            print('Upper bound updated itertivly to: ' + str(upper_bound))
    out_dict = {vector_idx:{'distances':distances, 'indices':indices}}        
             
    if vector_idx%1000==0 and vector_idx>0:
        dateTimeObj = datetime.now()
        timeObj = dateTimeObj.time()
        print('{} | index = {}, finished 1000 vectors'
              .format(timeObj, str(vector_idx)))
    out_q.put(out_dict)


def input_iterator(kd_tree, vectors, cluster_size, p, upper_bound):
    for idx, vector in enumerate(vectors):
        yield(kd_tree, vector, idx, cluster_size, p, upper_bound)


def worker_pavel(input_arg):
    (kd_tree, vector, idx, cluster_size, p, upper_bound) = input_arg
    
    assert isinstance(kd_tree, cKDTree)

    t0 = time.time()
    distances, indices = kd_tree.query(vector, cluster_size, p, distance_upper_bound=upper_bound)
    logger.info(str(datetime.now()) + '| kd_tree.query completed. Took {}'.format(time.time() - t0))

    while math.inf in distances:
            upper_bound = upper_bound * 1.1
            t0 = time.time()
            distances, indices = kd_tree.query(vector, cluster_size, p, distance_upper_bound=upper_bound)
            logger.info(str(datetime.now()) + '| kd_tree.query after upper_bound update completed. '
                                              'Took {}'.format(time.time() - t0))
            print('Upper bound updated itertivly to: ' + str(upper_bound))
    out_dict = {idx: {'distances': distances, 'indices': indices} }
    return out_dict


def get_proximity_list_parallel(data, cluster_size, cpus=2):

    logger.info(f'{datetime.now()} Started {get_proximity_list_parallel.__name__}'
                f' data length: {len(data)}. Using {cpus} cpus')
    t0 = time.time()
    kd_tree = cKDTree(data)
    logger.info(str(datetime.now()) + '| cKDtree ready. Took {}'.format(time.time() - t0))

    # initialize pool
    giant_result = {}
    upper_bound = 10.0
    t0 = time.time()
    
    with multiprocessing.Pool(processes=cpus) as pool:
        for res in pool.imap_unordered(func=worker_pavel, 
                                       iterable=input_iterator(kd_tree, data, cluster_size+1, 2, upper_bound), 
                                       chunksize=200):
            giant_result.update(res)
            if len(giant_result) % 1000 == 0:
                logger.info(f'{datetime.now()} finished {len(giant_result)} results so far')

    logger.info("{} finished running {}. Received {} results. Took {}"
                .format(datetime.now(), get_proximity_list_parallel.__name__, len(giant_result), time.time() - t0))
    
    proximity_list = np.zeros([data.shape[0], cluster_size+1])
    distances_list = np.zeros([data.shape[0], cluster_size+1])
    for idx, res_dict in giant_result.items():    
        proximity_list[idx] = np.array(res_dict['indices'], dtype=np.int)
        distances_list[idx] = np.array(res_dict['distances'], dtype=np.float)
    
    return proximity_list, distances_list


def cluster(data_file, output_file_path, output_file_name, cluster_size=100, cpus=2):
    """Cluster the data and write the result to output file"""
    vectors = read_vector_data(data_file)

    # open for debug
    #proximity, distances = get_proximity_list(vectors, cluster_size)
    proximity, distances = get_proximity_list_parallel(vectors, cluster_size, cpus)
    
    write_proximity_file(os.path.join(output_file_path, 'NN_'+output_file_name), proximity)
    write_proximity_file(os.path.join(output_file_path, 'Distances_'+output_file_name), distances)
    print(str(datetime.now()) + ' | finished clustering, files saved to: ')
    logger.info(str(datetime.now()) + ' | finished clustering, files saved to: ')
    logger.info(os.path.join(output_file_path, 'NN_' + output_file_name))
    logger.info(os.path.join(output_file_path, 'Distances_'+output_file_name))
    return proximity


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


def analyze_data_worker(input_arg):

    data, idx, row, fields_to_extract, id_field, subject_status, status_types = input_arg

    assert isinstance(data, pd.DataFrame)
    assert len(row) == 101
    assert isinstance(id_field, str)

    cluster_data = data.loc[row, fields_to_extract]
    subjects = cluster_data[id_field].unique()
    stats = get_subject_stats(subjects, subject_status, status_types)
    res = {
        'neighbors': [int(x) for x in row],
        'how_many_subjects': len(subjects),
    }
    res.update(stats)
    
    return idx, res


def analyze_data_parallel_input_generator(data, neighbors_list, fields_to_extract, id_field, subject_status, status_types):
    for idx, row in enumerate(neighbors_list):
        yield (data, idx, row, fields_to_extract, id_field, subject_status, status_types)


def analyze_data_parallel(neighbors_list, data_file, id_field='FILENAME', status_field='labels', cpus=4):
    """Return a dataframe with """
    logger.info(f'{str(datetime.now())} | Begin data analyze of nearest neighbors')
    data = pd.read_csv(data_file, sep='\t')
    # print(data.head())
    status_types = data[status_field].unique()
    subject_status = build_subject_status(data, id_field, status_field)
    logger.info(f'{str(datetime.now())} | Build subject status complete')
    list_out = [np.nan] * len(neighbors_list)
    fields_to_extract = [id_field, status_field]
    t0 = time.time()
    t1 = t0
    results = 0

    with multiprocessing.Pool(processes=cpus) as pool:
        for res in pool.imap_unordered(func=analyze_data_worker,
                iterable=analyze_data_parallel_input_generator(data, neighbors_list, fields_to_extract,
                                                               id_field, subject_status, status_types),
                chunksize=200):
            list_out[res[0]] = res[1]
            results +=1
            if results % 1000 == 0:
                logger.info(f'{datetime.now()} finished {results} results so far. time for 1000: {time.time() -t1} Elapsed time={time.time() -t0}')
                t1 = time.time()

    output_df = pd.DataFrame(list_out, columns=['neighbors', 'how_many_subjects'].extend(status_types))
    logger.info(f'{str(datetime.now())} | Finished analysis and transfered to dataframe')
    return output_df


def analyze_data(neighbors_list, data_file, id_field='FILENAME', status_field='labels'):
    """Return a dataframe with """
    logger.info(f'{str(datetime.now())} | Begin data analyze of nearest neighbors')
    data = pd.read_csv(data_file, sep='\t')
    #print(data.head())
    status_types = data[status_field].unique()
    subject_status = build_subject_status(data, id_field, status_field)
    logger.info(f'{str(datetime.now())} | Build subject status complete')
    list_out = [np.nan]*len(neighbors_list)
    fields_to_extract = [id_field, status_field]
    t0 = time.time()
    for idx, row in enumerate(neighbors_list):

        cluster_data = data.loc[row, fields_to_extract]
        subjects = cluster_data[id_field].unique()
        stats = get_subject_stats(subjects, subject_status, status_types)
        res = {
            'neighbors': [int(x) for x in row],
            'how_many_subjects': len(subjects),
        }
        res.update(stats)
        list_out[idx] = res
#        output.append(res)
        #print(output)
        #print(output_df)
        if idx % 1000 == 0:
            logger.info('{} | index = {}, finished 1000 vectors in {:.3} sec'.format(str(datetime.now()), str(row[:6]), time.time()-t0))
            t0 = time.time()

    output_df = pd.DataFrame(list_out,columns = ['neighbors', 'how_many_subjects'].extend(status_types))
    logger.info(f'{str(datetime.now())} | Finished analysis and transfered to dataframe')
    return output_df


def donttest_proximity_list_large():

    TEST_SIZE = 1000
    DIM = 10
    CLUSTER_SIZE = 100

    data = np.random.rand(TEST_SIZE, DIM)
    proximity = get_proximity_list(data, 100)
    assert proximity.shape == (TEST_SIZE, CLUSTER_SIZE)


def _generate_random_data(data_filename, data_size, data_dim):
    with open(data_filename, 'w', newline='') as fd:
        writer = csv.writer(fd)
        writer.writerow(range(data_dim))
        for _ in range(data_size):
            writer.writerow(np.random.rand(data_dim))


def test_read_vector_data():

    data_filename = '/home/miri-o/Desktop/test_data.csv'
    data_size = 10000
    data_dim = 100

    # build file if doesn't exist
    if not os.path.exists(data_filename):
        _generate_random_data(data_filename, data_size, data_dim)

    data = read_vector_data(data_filename)
    assert data.shape == (data_size, data_dim), "Wrong data size in reading data"


def test_cluster_small_data():
    print('Starting test')
    data_size = 4
    data_dim = 3
    data_filename = '/home/miri-o/Desktop/test_small_data.csv'
    output_file = '/home/miri-o/Desktop/res.csv'

    _generate_random_data(data_filename, data_size, data_dim)
    cluster(data_filename, output_file, 2)


    
if __name__ == '__main__':
    date_time_obj()
    # test_proximity_list()
    # options = parse_args()
#    test_analyze_data()
#    test_celiac_data_10K()

