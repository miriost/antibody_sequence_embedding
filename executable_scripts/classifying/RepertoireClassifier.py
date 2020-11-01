from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone

class RepertoireClassifier:

    def __init__(self, name, estimator=None, trained_model=None, feature_selector=None, features=None, parameters=None):
        self.estimator = estimator
        self.name = name
        self.feature_selector = feature_selector
        self.trained_model = trained_model
        self.features = features
        self.parameters = parameters

    def select_features(self, X_train, y_train, n_splits=20, repeated=True):

        if self.feature_selector is not None:
            estimator = clone(self.estimator)
            self.features = self.feature_selector(estimator, X_train, y_train, n_splits, repeated)
        else:
            self.features = X_train.columns.to_list()

    def fit(self, X_train, y_train, n_splits=20, repeated=True):

        X_train = X_train.loc[:, self.features]
        estimator = clone(self.estimator)
        self.trained_model = GridSearchCV(estimator,
                                          self.parameters,
                                          cv=KFold(n_splits=n_splits, shuffle=repeated, random_state=0),
                                          scoring='f1_weighted')
        self.trained_model.fit(X_train, y_train)
        self.parameters = self.trained_model.best_params_

    def predict(self, X_test):

        X_test = X_test.loc[:, self.features]
        return self.trained_model.predict(X_test)
