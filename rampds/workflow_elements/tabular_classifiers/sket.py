import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.base import BaseEstimator
from ramphy import Hyperparameter

# RAMP START HYPERPARAMETERS
n_estimators = Hyperparameter(dtype='int', default=100, values=[10, 20, 30, 40, 50, 70, 100, 150, 200, 250, 300, 400, 500, 700, 1000])
max_depth = Hyperparameter(dtype='int', default=5, values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100])
min_samples_split = Hyperparameter(dtype='int', default=2, values=[2, 5, 10, 20, 50, 100, 200, 500, 700])
min_data_in_leaf = Hyperparameter(dtype='int', default=1, values=[1, 2, 4, 6, 8, 10, 20, 50])
colsample_bytree = Hyperparameter(dtype='float', default=0.5, values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
bootstrap = Hyperparameter(dtype='bool', default=True, values=[True, False])
criterion = Hyperparameter(dtype='str', default='gini', values=['gini', 'entropy', 'log_loss'])
min_weight_fraction_leaf = Hyperparameter(dtype='float', default=0.0, values=[0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5])
min_impurity_decrease = Hyperparameter(dtype='float', default=0.0, values=[0.0, 0.0001, 0.001, 0.01])
ccp_alpha = Hyperparameter(dtype='float', default=0.0, values=[0.0, 0.0001, 0.001, 0.01, 0.1])
max_samples = Hyperparameter(dtype='float', default=-1.0, values=[-1.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
# RAMP END HYPERPARAMETERS

N_ESTIMATORS = int(n_estimators)
MAX_DEPTH = int(max_depth)
MIN_SAMPLES_SPLIT = int(min_samples_split)
MIN_SAMPLES_LEAF = int(min_data_in_leaf)
MAX_FEATURES = float(colsample_bytree)
BOOTSTRAP = bool(bootstrap)
CRITERION = str(criterion)
MIN_WEIGHT_FRACTION_LEAF = int(min_weight_fraction_leaf)
MIN_IMPURITY_DECREASE = float(min_impurity_decrease)
CCP_ALPHA = float(ccp_alpha)
MAX_SAMPLES = None if not BOOTSTRAP or float(max_samples) == -1.0 else float(max_samples)

class Classifier(BaseEstimator):
    def __init__(self, metadata):
        self.metadata = metadata
        target_cols = metadata["data_description"]["target_cols"]
        if len(target_cols) > 1:
            raise NotImplementedError("Multi-output classification is not yet supported.")

    def fit(self, X, y):
        self.clf = ExtraTreesClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            min_samples_split=MIN_SAMPLES_SPLIT,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            max_features=MAX_FEATURES,
            bootstrap=BOOTSTRAP,
            criterion=CRITERION,
            min_weight_fraction_leaf=MIN_WEIGHT_FRACTION_LEAF,
            min_impurity_decrease=MIN_IMPURITY_DECREASE,
            ccp_alpha=CCP_ALPHA,
            max_samples=MAX_SAMPLES,
        )
        self.clf.fit(X, y.ravel())

    def predict_proba(self, X):
        y_proba = self.clf.predict_proba(X)
        return y_proba
