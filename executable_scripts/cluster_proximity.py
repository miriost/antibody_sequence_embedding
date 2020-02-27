import os
import csv
import argparse
from collections import defaultdict

import pandas as pd
import numpy as np
from scipy.spatial import KDTree
import time

def parse_args():
    pass


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

    # Build KDTree to inquire the closest points
    tree = KDTree(data)
    print('KDtree ready')
    t0 = time.time()
    for vector_idx, vector in enumerate(data):
        # the query will always find vector as nearest result
        
        distances, indices = tree.query(vector, cluster_size + 1)
        
        if vector_idx%100==0:
            print('index = {}, finished 100 vectors in {:.3} sec'.format(str(vector_idx), time.time()-t0))
            t0 = time.time()
#        print('vector index {}, closest to {}, distances {}:'.format(vector_idx, indices, distances[:2]))

        #assert indices[0] == vector_idx, "First index should be the vector itself"
        #indices = indices[1:]
        proximity_list[vector_idx] = np.array(indices, dtype=np.int)

    return proximity_list


def cluster(data_file, output_file, cluster_size=100):
    """Cluster the data and write the result to output file"""
    vectors = read_vector_data(data_file)
    proximity = get_proximity_list(vectors, cluster_size)
    write_proximity_file(output_file, proximity)
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

    output = [] # pd.DataFrame(columns=['neighbors', 'how_many_subjects'].extend(status_types))

    fields_to_extract = [id_field, status_field]
    for row in neighbors_list:
        cluster_data = data.loc[row, fields_to_extract]
        subjects = cluster_data[id_field].unique()
        stats = get_subject_stats(subjects, subject_status, status_types)

        res = {
            'neighbors': row,
            'how_many_subjects': len(subjects),
        }
        res.update(stats)

        output.append(res)
        #print(output)
        output_df = pd.DataFrame(output)
        #print(output_df)
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


def test_analyze_data():
    """Test main analyze function"""
    if False: #100K head of the whole celiac data set
        data_file_path = '/home/miri-o/Desktop/test_100000_data.csv'
        vectors_file_path = '/home/miri-o/Desktop/test_100000_vectors.csv'
        output_file_path = '/home/miri-o/Desktop/clusters.csv'
        neighbors_list = cluster(vectors_file_path, output_file_path, cluster_size=100)
        print(neighbors_list)
        out = analyze_data(neighbors_list, data_file_path)
        output_file = '/home/miri-o/Desktop/res_100000_test.csv'
        out.to_csv(output_file)
    if True: #160K samples, from only CI and SC subjects with >25K sequences
        data_file_path = '/media/miri-o/Documents/filtered_data_sets/HCV_data_only_CI_SC_10K_per_subject_CDR3_from_Celiac_trimmied_3_4_FILTERED_DATA.csv'
        
        vectors_file_path = '/media/miri-o/Documents/vectors/HCV_data_only_CI_SC_10K_per_subject_CDR3_from_Celiac_trimmied_3_4_VECTORS.csv'
        output_file_path = '/home/miri-o/Desktop/clusters_HCV_10K_per_subject_CDR3_from_Celiac_trimmied_3_4_VECTORS.csv'
        neighbors_list = cluster(vectors_file_path, output_file_path, cluster_size=100)
        print(neighbors_list)
        out = analyze_data(neighbors_list, data_file_path)
        output_file = '/home/miri-o/Desktop/NN_results_HCV_10K_per_subject_Celiac_model_trimmied_3_4_VECTORS.csv'
        out.to_csv(output_file)
        

if __name__ == '__main__':
    pass
    # test_proximity_list()
    # options = parse_args()
    test_analyze_data()

