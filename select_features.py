import pandas as pd
import os
import sys
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel


def main():
    parser = argparse.ArgumentParser(description='use random forest feature selector to select important features from'
                                                 'the feature_table')
    parser.add_argument('train_feature_file', help='the train feature table csv file')
    parser.add_argument('test_feature_file', help='the test feature table csv file')
    parser.add_argument('--max_features', help='the maximum number of features, default is 300', type=int, deafult=300)
    parser.add_argument('--threshold', help='A scaling factor for the feature selection importance threshold (e.g., '
                                            '“factor*mean_importance”)', type=float, default=1.0)

    args = parser.parse_args()
    execute(args)


def execute(args):

    train_feature_file = args.train_feature_file
    test_feature_file = args.test_feature_file
    max_features = args.max_features
    threshold = args.threshold

    if not(os.path.isfile(train_feature_file)):
        print('train feature file error, make sure file path exists\nExiting...')
        sys.exit(1)
    train_feature_table = pd.read_csv(train_feature_file)

    if not(os.path.isfile(test_feature_file)):
        print('train feature file error, make sure file path exists\nExiting...')
        sys.exit(1)
    test_feature_table = pd.read_csv(train_feature_file)

    X_train = train_feature_table.drop(columns=['subject.disease_diagnosis'])
    y_train = train_feature_table['subject.disease_diagnosis']

    sel = SelectFromModel(RandomForestClassifier(), max_features=max_features, threshold=threshold)
    sel.fit(X_train, y_train)
    selected_features = X_train.columns[(sel.get_support())].tolist()

    print('selected {} features from {} original features'.format(len(selected_features), X_train.shpae[0]))

    base_name = os.path.basename(train_feature_file).split['.csv'][0]
    dir_name = os.path.dirname(train_feature_file)
    full_path = os.path.join(dir_name, 'selected_' + base_name + '.csv')
    train_feature_table[selected_features + ['subject.disease_diagnosis']].to_csv(os.path.join(full_path))
    print('saved {}'.format(full_path))

    base_name = os.path.basename(test_feature_file).split['.csv'][0]
    dir_name = os.path.dirname(test_feature_file)
    full_path = os.path.join(dir_name, 'selected_' + base_name + '.csv')
    test_feature_table[selected_features + ['subject.disease_diagnosis']].to_csv(os.path.join(full_path))
    print('saved {}'.format(full_path))


if __name__ == '__main__':
    main()
