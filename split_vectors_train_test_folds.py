import sys, argparse, os
import pandas as pd
import numpy as np


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', "True"):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', "False"):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():

    parser = argparse.ArgumentParser(description='Split a data file to train and test data sets')

    parser.add_argument('data_file_path', help='The original tsv data file')
    parser.add_argument('vectors_file_path', help='A vector file to split according to the data files split')
    parser.add_argument('--folds_dir', help='A root dir containing the data folds, each fold is in a directory '
                                            'FOLD{fold number}, default is "./"', default='./')
    parser.add_argument('--n_splits',  help='Number of splits, default=10', type=int, default=10)
    args = parser.parse_args()

    execute(args)


def execute(args):

    vectors_file_path = args.vectors_file_path
    folds_dir = args.folds_dir
    n_splits = args.n_splits
    data_file_path = args.data_file_path

    if not os.path.isdir(folds_dir):
        print('folds dir error! Make sure the directory exists.\nExiting...')
        sys.exit(1)

    if not os.path.isfile(data_file_path):
        print('data file error! Make sure the file exists and it is *.tsv file.\nExiting...')
        sys.exit(1)
    data_file = pd.read_csv(data_file_path, sep='\t')
    print('Vectors file loaded, original file length: ', data_file.shape[0])

    if not os.path.isfile(vectors_file_path):
        print('vectors file error! Make sure the file exists and it is *.npy file.\nExiting...')
        sys.exit(1)
    vectors_file = np.load(vectors_file_path)
    print('Vectors file loaded, original file length: ', vectors_file.shape[0])

    if vectors_file.shape[0] != data_file.shape[0]:
        print('mismatch between data_file and vectors_file length\nExiting...')
        sys.exit(1)

    data_file_desc = os.path.basename(data_file_path).split('.tsv')[0]
    vectors_file_desc = os.path.basename(vectors_file_path).split('_VECTORS.npy')[0]
    for fold in range(n_splits):
        for split in ['TRAIN_', 'TEST_']:
            file_name = os.path.join(folds_dir, 'FOLD' + str(fold), data_file_desc + '_' + split + '.tsv')
            if not os.path.isfile(file_name):
                print('{} file error! Make sure the file exists.\nSkipping...'.format(file_name))
                continue
            sequences_ids = pd.read_csv(file_name, sep='\t', usecols=['document._id'])
            indexing = data_file['document._id'].isin(sequences_ids['document._id'])
            vectors = vectors_file[indexing, :]
            vectors_file_name = os.path.join(folds_dir, 'FOLD' + str(fold), vectors_file_desc +'_VECTORS_' + split +
                                             '.npy');
            print('saving: ', vectors_file_name)
            np.save(vectors_file_name, vectors)


if __name__ == "__main__":
    main()
