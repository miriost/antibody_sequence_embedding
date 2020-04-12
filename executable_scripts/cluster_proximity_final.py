import os
import csv
import argparse
from collections import defaultdict
import math

import pandas as pd
import numpy as np
from scipy.spatial import KDTree
import time
import sys
#import pprofile
#import cProfile
from datetime import datetime

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file_path', 
                        help='the filtered data file path')
    parser.add_argument('vectors_file_path', 
                        help='the vectors file path')
    parser.add_argument('output_folder_path', 
                        help='Output folder for the 3 output files - Nearest neighbors file, destances file, and results anaylsis file')
    parser.add_argument('output_description', 
                        help='description to use inside output file names')
    args = parser.parse_args()
    if not(os.path.isfile(args.data_file_path)):
        print('feature file error, make sure feature and vectors file path\nExiting...')
        sys.exit(1)
        
    if not(os.path.isfile(args.vectors_file_path)):
        print('vectors file error, make sure feature and vectors file path\nExiting...')
        sys.exit(1)
        
    if not os.path.exists(args.output_folder_path):
        os.mkdir(args.output_folder_path)
        print('output_folder_created: ' + str(args.output_folder_path))
    
    dateTimeObj = datetime.now()
    timeObj = dateTimeObj.time()
    print(timeObj, 'Beginning full data analysis')
    print('Data file path:' + args.data_file_path)
    print('Vectors file path:' + args.vectors_file_path)
    print('-----------------------')
    output_file_name = args.output_description + '.csv'
    neighbors_list = cluster(args.vectors_file_path, args.output_folder_path, output_file_name, cluster_size=100)
#    neighbors_list = np.loadtxt(output_file_path, delimiter=',', skiprows =1)
    out = analyze_data(neighbors_list, args.data_file_path)
    output_file = args.output_description + '_analysis.csv'
    out.to_csv(os.path.join(args.output_folder_path, output_file))
    print(str(datetime.now()) +  ' | data written to output file: ' + output_file)
    
def read_vector_data(input_file):
    """Read vector data from input file and return it as pandas vector list"""
    from numpy import genfromtxt
    my_data = genfromtxt(input_file, delimiter=',', skip_header=1)
    return my_data


def write_proximity_file(output_file, proximity):
    header = ', '.join(str(el) for el in range(proximity.shape[1]))
    np.savetxt(output_file, proximity, delimiter=',', header=header)


def get_proximity_list(data, cluster_size):
    proximity_list = np.zeros([data.shape[0], cluster_size+1])
    distances_list = np.zeros([data.shape[0], cluster_size+1])
#    profiler = pprofile.Profile()
#    with profiler:
        # Build KDTree to inquire the closest points
    tree = KDTree(data)
    print(str(datetime.now()) + '| KDtree ready')
    t0 = time.time()
    upper_bound = 10
    for vector_idx, vector in enumerate(data):
        # the query will always find vector as nearest result        
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
            print('{} | index = {}, finished 1000 vectors in {:.3} sec'.format(timeObj, str(vector_idx), time.time()-t0))
            #distances_average = np.mean(distances_list[(vector_idx-1000):(vector_idx+1)])
            distances_max = np.max(distances_list[(vector_idx-1000):(vector_idx+1)])
            #print('average distance between neighbors, indexes {} to {}: {}'.format((vector_idx-1000),(vector_idx+1), distances_average))
            print('Maximal distance between neighbors, indexes {} to {}: {}'.format((vector_idx-1000),(vector_idx+1), distances_max))
            upper_bound = (upper_bound*(vector_idx//1000) + distances_max)/(1+vector_idx//1000)
            print('Updating upper bound to:' + str(upper_bound))
            t0 = time.time()
#        print('vector index {}, closest to {}, distances {}:'.format(vector_idx, indices, distances[:2]))
        #assert indices[0] == vector_idx, "First index should be the vector itself"
        #indices = indices[1:]
#        profiler.print_stats() 
    return proximity_list, distances_list


def cluster(data_file, output_file_path, output_file_name, cluster_size=100):
    """Cluster the data and write the result to output file"""
    vectors = read_vector_data(data_file)
    proximity, distances = get_proximity_list(vectors, cluster_size)
    write_proximity_file(os.path.join(output_file_path, 'NN_'+output_file_name), proximity)
    write_proximity_file(os.path.join(output_file_path, 'Distances_'+output_file_name), distances)
    print(str(datetime.now()) + ' | finished clustering, files saved to: ')
    print(os.path.join(output_file_path, 'NN_'+output_file_name))
    print(os.path.join(output_file_path, 'Distances_'+output_file_name))
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


def analyze_data(neighbors_list, data_file, id_field='FILENAME', status_field='labels'):
    """Return a dataframe with """

    data = pd.read_csv(data_file)
    print(data.head())
    status_types = data[status_field].unique()
    subject_status = build_subject_status(data, id_field, status_field)
    print('build subject status complete')
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
        if int(row[0])%1000==0:
            print(str(datetime.now()) + ' | index = {}, finished 1000 vectors in {:.3} sec'.format(str(row), time.time()-t0))
            t0 = time.time()
    output_df = pd.DataFrame(list_out,columns = ['neighbors', 'how_many_subjects'].extend(status_types))
    return output_df

def test_proximity_list_small():

    d1 = np.array([[0., 0., 0.],
                   [1., 0., 0.],
                   [0., 0., 1.],
                   [0.1, 0.1, 0.1]])
    proximity = get_proximity_list(d1, 2)
    assert 3 in proximity[0]
    assert 0 in proximity[1]
    assert 0 in proximity[2]
    assert 0 in proximity[3]


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
    main(sys.argv[1:])
    # test_proximity_list()
    # options = parse_args()
#    test_analyze_data()
#    test_celiac_data_10K()

