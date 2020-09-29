import sys, argparse
import os
import matplotlib.pyplot as plt
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()).split('antibody_sequence_embedding')[0])            
from antibody_sequence_embedding.classifier import classifier
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


def main(argv):
    parser = argparse.ArgumentParser(
        description='''classify_no_splitting.py function performs classification based on a train and test feature tables where each row is an observation and each column is a feature. It's output is a confusion matrix and a score.  ''',
        epilog="""All's well that ends well.""")
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', help='a *.csv file containing the TRAIN features table, including labels column')
    parser.add_argument('--test_file', help='a *.csv file containing the TEST features table, including labels column')
    parser.add_argument('--labels_col_name', help='Name of the labels column, default: "labels"', default='labels')
    parser.add_argument('-M', '--model',
                        help='Classification model. current supported models: logistic_regression, decision_tree, kNN, linear_svm, RBF_SVM, Gaussian, Random_Forest, MLP, ADA, MLP, naive_bayes, QDA" , default: "decision_tree"',
                        default="decision_tree")

    if len(sys.argv) <= 1:
        sys.argv.extend(r"--train_file C:\Users\mirio\research\cross_validation\10K_TRAIN_0_S_60_feature_table_with_labels.csv"
                        r" --test_file C:\Users\mirio\research\cross_validation\10K_TEST_0_S_60_feature_table_with_labels.csv -M all".split())

    args = parser.parse_args()

    if not (os.path.isfile(args.train_file) and os.path.splitext(os.path.split(args.train_file)[1])[1] == '.csv'):
        print('train file {} error! Make sure the file exists and it is *.csv file.\nExiting...'.format(
            args.train_file))
        sys.exit(1)
    train_file = pd.read_csv(args.train_file, index_col = 0)
    if not (os.path.isfile(args.test_file) and os.path.splitext(os.path.split(args.test_file)[1])[1] == '.csv'):
        print('test file {} error! Make sure the file exists and it is *.csv file.\nExiting...'.format(
            args.test_file))
        sys.exit(1)
    test_file = pd.read_csv(args.test_file, index_col = 0)


    if args.labels_col_name:
        if (not args.labels_col_name in train_file.columns) or (not args.labels_col_name in test_file.columns):
            print('label column name doesnt exist.\nExiting...')
            sys.exit(1)
        else:
            labels_col_name = args.labels_col_name
    else:
        labels_col_name = 'labels'

    print(
        "~~~~~~~~ Begin classifing...\nTrain file: {}\nTest file: {}\nLabels column name: {}\nModel: {}\n~~~~~~~".format(
            os.path.abspath(args.train_file), os.path.abspath(args.test_file), labels_col_name, args.model))

    x_train = train_file.drop(labels_col_name, axis = 1)
    y_train = train_file[labels_col_name]
    x_test = test_file.drop(labels_col_name, axis=1)
    y_test = test_file[labels_col_name]

    labels_all = pd.concat([y_train, y_test])

    feature_table = pd.concat([x_train, x_test])
    if args.model == 'all':
        models = ["logistic_regression", "decision_tree", "kNN", "linear_svm", "RBF_SVM", "Gaussian",
                  "Random_Forest", "MLP", "ADA", "MLP", "naive_bayes", "QDA"]
        for mod in models:
            our_classifier = classifier(feature_table=feature_table, labels=labels_all, model=mod, C=.9)
            our_classifier.run_once(x_train, x_test, y_train, y_test)
            print(f'original test labels: {list(y_test)}')
    else:

        our_classifier = classifier(feature_table= feature_table, labels=labels_all, model = args.model, C=.9)
        our_classifier.run_once(x_train, x_test, y_train, y_test)

if __name__ == "__main__":
    main(sys.argv[1:])
