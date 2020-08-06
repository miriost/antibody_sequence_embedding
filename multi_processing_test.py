# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 21:11:02 2020

@author: mirio
"""
#multi-proceesing test
import time
import multiprocessing 
import random

def basic_func(x):
    if x == 0:
        return 'zero'
    elif x%2 == 0:
        return 'even'
    else:
        return 'odd'

def multiprocessing_func(x,z,a):
    print(x,z)
    y = x*z
    time.sleep(2)
    print('{}*{} results in a/an {} number'.format(x, z, basic_func(y)))
    a[x] = y
    
if __name__ == '__main__':
    starttime = time.time()
#    processes = []
    l = [random.randint(0,15) for x in range(0,10)]
    arr = multiprocessing.Array('i', range(10))
    print(arr[:])
    for i, num in enumerate(l):
        p = multiprocessing.Process(target=multiprocessing_func, args=(i,num,arr))
#        processes.append(p)
        p.start()
        p.join()
        
#    for process in processes:
#        process.join()
    print(arr[:])    
    print('That took {} seconds'.format(time.time() - starttime))
    
## pavels tests
# backup - this function works!
def input_iterator_backup(kd_tree, vectors):
    for idx, vector in enumerate(vectors):
        yield (kd_tree, vector, idx)

def fake_worker(input_arg):
    (kd_tree, vector, idx) = input_arg
    assert isinstance(kd_tree, KDTree)
    assert len(vector) == 100
    assert isinstance(idx, int)

    vectors = list(range(17))
    distances = list(range(-3, 17, 2))

    return (idx, vectors, distances)

def get_proximity_list_parallel_backup(data, cluster_size):

    MAX_PROCESSES = 4
    giant_result = {}
    kd_tree = KDTree(data)
    # vectors = 30 * [list(range(100))]

    print(f'data length: {len(data)}')
    print(str(datetime.now()) + '| KDtree ready')

    with multiprocessing.Pool(processes=MAX_PROCESSES) as pool:
        for res in pool.imap_unordered(func=fake_worker, 
                                       iterable=input_iterator(kd_tree, data), 
                                       chunksize=1000):
            (idx, vectors, distances) = res
            giant_result[idx] = {'vectors': vectors, 'distances': distances}

            if len(giant_result) % 100 == 0:
                print(f'{datetime.now()} got some {len(giant_result)} results there bitches')

    print(f"{datetime.now()} finished. Received {len(giant_result)} results")

