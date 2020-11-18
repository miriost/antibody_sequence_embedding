import sys, argparse, os
from random import choices
import pandas as pd
import pathlib
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


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
    parser.add_argument('data_file', help='A file containing raw of data, including a label and subject id columns')
    parser.add_argument('--n_splits',  help='Number of CV splits, default=5', type=int, default=5)
    parser.add_argument('--n_repeats', help='Number of CV repeats, default=1', type=int, default=1)
    parser.add_argument('--test_size', help='The proportion of the data set, default=0.2', type=float, default=0.2)
    parser.add_argument('--shuffle_labels', help='Create the splits with shuffled labels. Default=False',
                        type=str2bool, default=False)
    parser.add_argument('--output_dir',
                        help='Output directory where the folds directories will be created, default: "./"',
                        default='./')
    args = parser.parse_args()

    execute(args)


def execute(args):

    if not os.path.isfile(args.data_file):
        print('feature file error! Make sure the file exists and it is *.csv file.\nExiting...')
        sys.exit(1)
    data_file = pd.read_csv(args.data_file, sep='\t')
    print('Data file loaded, original file length: ', str(len(data_file)))

    label_column = 'subject.disease_diagnosis'
    if not (label_column in data_file.columns):
        print(data_file.columns)
        print(label_column + ' column is missing in data file\nExiting...')
        sys.exit(1)

    id_column = 'subject.subject_id'
    if id_column not in data_file.columns:
        print(data_file.columns)
        print(id_column + ' column is missing in data file\nExiting...')
        sys.exit(1)

    n_repeats = args.n_repeats
    n_splits = args.n_splits
    test_size = args.test_size
    shuffle_labels = args.shuffle_labels

    subjects = data_file.loc[:, ].groupby(by=[id_column])[label_column].apply(lambda x: x.iloc[0])

    if shuffle_labels is True:
        print("shuffling labels before split")
        subjects[:] = np.random.permutation(subjects)
        for subject_id, label in subjects.iteritems():
            data_file.loc[data_file[id_column] == subject_id, label_column] = label

    fold = 0
    for random_state in range(n_repeats):

        kf = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)

        for train, test in kf.split(subjects.index, subjects):

            train_subjects = subjects.iloc[train]
            train_output = data_file.loc[data_file[id_column].isin(train_subjects.index), :]

            test_subjects = subjects.iloc[test]
            test_output = data_file.loc[data_file[id_column].isin(test_subjects.index), :]

            output_dir = os.path.join(args.output_dir, 'FOLD' + str(fold))
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
            file_name = os.path.basename(args.data_file).replace('.tsv', '')
            train_output.to_csv(os.path.join(output_dir, file_name + '_TRAIN_' + str(fold) + '.tsv'), sep='\t', index=False)
            test_output.to_csv(os.path.join(output_dir, file_name + '_TEST_' + str(fold) + '.tsv'), sep='\t', index=False)

            fold += 1

            print(f'----- fold {fold} files saved -----')


def test_create_folds():

    subjects = pd.DataFrame(columns=['subject.disease_diagnosis'], index=['S' + str(x) for x in range(20)])
    subjects['subject.disease_diagnosis'] = choices(['Sick', 'Healthy'], k=20)

    data_file = pd.DataFrame()
    data_file['subject.subject_id'] = choices(subjects.index, k=100)
    data_file['subject.disease_diagnosis'] = [subjects.loc[x, 'subject.disease_diagnosis']
                                              for x in data_file['subject.subject_id']]

    data_file.to_csv('test_create_folds.tsv', sep='\t', index_label='index')

    class DummyArgs:

        def __init__(self):
            self.data_file = 'test_create_folds.tsv'
            self.n_splits = 5
            self.n_repeats = 2
            self.test_size = 0.2
            self.output_dir = './'
            self.shuffle_labels = False

    args = DummyArgs()

    execute(args)


if __name__ == "__main__":
    main()

