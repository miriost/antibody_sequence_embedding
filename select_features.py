import pandas as pd
import os
import sys
import argparse
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def main():
    parser = argparse.ArgumentParser(description='use random forest feature selector to select important features from'
                                                 'the feature_table')
    parser.add_argument('train_feature_file', help='the train feature table csv file')
    parser.add_argument('test_feature_file', help='the test feature table csv file')
    parser.add_argument('--max_features', help='the maximum number of features, default is 300', type=int, default=300)
    parser.add_argument('--threshold', help='A scaling factor for the feature selection importance threshold (e.g., '
                                            '“factor*mean_importance”)', type=float, default=1.0)
    parser.add_argument('--normalize_rows', help='normalize the rows (divide by row sum). Default is True.',
                        type=str2bool, default=True)
    parser.add_argument('--scale_columns', help='use min-max scaler to scale rows, Default is False', type=str2bool,
                        default=False)
    parser.add_argument('--selection_method', help='which cross validation method to use, options are: '
                                                   'rfecv, SelectFromModel, custom. Default is custom.', type=str,
                        default='custom')

    args = parser.parse_args()
    execute(args)


def execute(args):

    train_feature_file = args.train_feature_file
    test_feature_file = args.test_feature_file
    max_features = args.max_features
    threshold = args.threshold
    normalize_rows = args.normalize_rows
    scale_columns = args.scale_columns
    label_column = 'subject.disease_diagnosis'
    subject_column = 'SUBJECT'
    selection_method = args.selection_method

    if not(os.path.isfile(train_feature_file)):
        print('train feature file error, make sure file path exists\nExiting...')
        sys.exit(1)
    train_feature_table = pd.read_csv(train_feature_file)

    if not(os.path.isfile(test_feature_file)):
        print('train feature file error, make sure file path exists\nExiting...')
        sys.exit(1)
    test_feature_table = pd.read_csv(train_feature_file)

    if normalize_rows:
        print('normalizing the rows')
        tmp = train_feature_table.loc[:, ~train_feature_table.columns.isin([subject_column, label_column])]
        train_feature_table.loc[:,  ~train_feature_table.columns.isin([subject_column, label_column])] = tmp.div(tmp.sum(axis=1), axis=0)

        tmp = test_feature_table.loc[:, ~test_feature_table.columns.isin([subject_column, label_column])]
        test_feature_table.loc[:,  ~test_feature_table.columns.isin([subject_column, label_column])] = tmp.div(tmp.sum(axis=1), axis=0)

    if scale_columns:
        scaler = MinMaxScaler()
        tmp = train_feature_table.loc[:, ~train_feature_table.columns.isin([subject_column, label_column])]
        scaler.fit(tmp)
        train_feature_table.loc[:, ~train_feature_table.columns.isin([subject_column,
                                                                      label_column])] = scaler.transform(tmp)

        scaler = MinMaxScaler()
        tmp = test_feature_table.loc[:, ~test_feature_table.columns.isin([subject_column, label_column])]
        scaler.fit(tmp)
        test_feature_table.loc[:, ~test_feature_table.columns.isin([subject_column,
                                                                    label_column])] = scaler.transform(tmp)

    y_train = train_feature_table[label_column]
    X_train = train_feature_table.drop(columns=[label_column, subject_column])

    selected_features = X_train.columns

    if selection_method == 'select_from_model':
        print('using select from model feature selection')
        sel = SelectFromModel(RandomForestClassifier(n_estimators=100), max_features=max_features, threshold=threshold)
        sel.fit(X_train, y_train)
        selected_features = X_train.columns[(sel.get_support())].tolist()

    elif selection_method == 'rfecv':
        print('using rfecv feature selection')
        rf = RandomForestClassifier()
        min_features_to_select = 1  # Minimum number of features to consider
        rfecv = RFECV(estimator=rf, step=1, cv=StratifiedKFold(5),
                      scoring='accuracy',
                      min_features_to_select=min_features_to_select)
        rfecv.fit(X_train, y_train)
        selected_features = X_train.columns[(rfecv.get_support())].tolist()

    elif selection_method == 'custom':
        print('using custom feature selection')
        data_x = X_train
        data_y = y_train
        kf = StratifiedKFold(n_splits=5, shuffle=True)
        N = round(len(X_train.columns) * 0.3)
        all_importances = pd.DataFrame()
        for train, _ in kf.split(data_x, data_y):
            rfc = RandomForestClassifier(n_estimators=100)
            rfc.fit(data_x.iloc[train, :], data_y[train])
            importances1 = pd.DataFrame(
                {'FEATURE': data_x.columns, 'IMPORTANCE': np.round(rfc.feature_importances_, 3)})
            importances = importances1.sort_values('IMPORTANCE', ascending=False).set_index('FEATURE')
            # appending
            all_importances = all_importances.append(importances)

        all_importances_mean = all_importances.groupby('FEATURE').mean()
        all_importances_sort = all_importances_mean.sort_values('IMPORTANCE', ascending=False)
        # saving
        average_importance_features = list(all_importances_sort.index)
        # ~~~~~~~~~~~~~~~~ FS_RFE ~~~~~~~~~~~~~~~~~~~~~~~
        rf_features = average_importance_features[0:N]
        X = X_train.loc[:, rf_features]
        model = LogisticRegression()
        rfe = RFE(model, n_features_to_select=round(N * 0.3), step=0.01)
        rfe = rfe.fit(X, y_train)

        # saving selected features
        index_feature = [i for i, x in enumerate(rfe.support_) if x]
        selected_features = X.columns.values[index_feature].to_list()

    selected_features.append(label_column)
    selected_features.append(subject_column)

    print('selected {} features from {} features'.format(len(selected_features), X_train.shape[0]))

    base_name = os.path.basename(train_feature_file).split('.csv')[0]
    dir_name = os.path.dirname(train_feature_file)
    full_path = os.path.join(dir_name, 'selected_' + base_name + '.csv')
    train_feature_table[selected_features].to_csv(os.path.join(full_path))
    print('saved {}'.format(full_path))

    base_name = os.path.basename(test_feature_file).split('.csv')[0]
    dir_name = os.path.dirname(test_feature_file)
    full_path = os.path.join(dir_name, 'selected_' + base_name + '.csv')
    test_feature_table[selected_features].to_csv(os.path.join(full_path))
    print('saved {}'.format(full_path))


if __name__ == '__main__':
    main()
