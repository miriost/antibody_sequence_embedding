from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


class RepertoireClassifier:

    def __init__(self, name, estimator=None, trained_model=None, feature_selector=None, features=None, parameters=None):
        self.estimator = estimator
        self.name = name
        self.feature_selector = feature_selector
        self.trained_model = trained_model
        self.features = features
        self.parameters = parameters
        self.scaler = None

    def select_features(self, X_train, y_train, n_splits=10, repeated=True):

        if self.feature_selector is None or (len(X_train.columns) < 100):
            self.features = X_train.columns
            return

        self.scaler = MinMaxScaler()
        self.scaler.fit(X_train)
        columns = X_train.columns 

        X_train = pd.DataFrame(self.scaler.transform(X_train), columns=columns)

        estimator = clone(self.estimator)
        self.features = self.feature_selector(estimator, X_train, y_train, n_splits, repeated)
        self.features = X_train.columns.to_list()

    def fit(self, X_train, y_train, n_splits=10, repeated=True):

        X_train = X_train.loc[:, self.features]

        if self.scaler is None:
            self.scaler = MinMaxScaler()
            self.scaler.fit(X_train)

        X_train = self.scaler.transform(X_train)

        estimator = clone(self.estimator)
        if (len(self.parameters) == 1) and (len(self.parameters[0]) == 0):
            n_splits = 2

        self.trained_model = GridSearchCV(estimator,
                                          self.parameters,
                                          cv=KFold(n_splits=n_splits, shuffle=repeated, random_state=0),
                                          scoring='f1_weighted')
        self.trained_model.fit(X_train, y_train)
        self.parameters = self.trained_model.best_params_

    def predict(self, X_test):

        X_test = X_test.loc[:, self.features]
        X_test = self.scaler.transform(X_test)

        return self.trained_model.predict(X_test)

