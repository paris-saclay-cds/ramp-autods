import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline                                                              
from sklearn.preprocessing import StandardScaler                                                        
from ramphy import Hyperparameter

# RAMP START HYPERPARAMETERS
n_neighbors = Hyperparameter(dtype='int', default=5, values=[1, 2, 3, 5, 7, 10, 15, 20, 25, 30])
weights = Hyperparameter(dtype='str', default='uniform', values=['uniform', 'distance'])
algorithm = Hyperparameter(dtype='str', default='kd_tree', values=['ball_tree', 'kd_tree'])
leaf_size = Hyperparameter(dtype='int', default=30, values=[10, 20, 30, 40, 50])
p = Hyperparameter(dtype='int', default=2, values=[1, 2])
metric = Hyperparameter(dtype='str', default='euclidean', values=['minkowski', 'euclidean', 'manhattan', 'chebyshev'])
# RAMP END HYPERPARAMETERS

N_NEIGHBORS = int(n_neighbors)
WEIGHTS = str(weights)
ALGORITHM = str(algorithm)
LEAF_SIZE = int(leaf_size)
P = int(p)
METRIC = str(metric)

class Classifier(BaseEstimator):
    def __init__(self, metadata):
        self.metadata = metadata
        target_cols = metadata["data_description"]["target_cols"]
        if len(target_cols) > 1:
            raise NotImplementedError("Multi-output classification is not yet supported.")

    def fit(self, X, y):
        self.clf = make_pipeline(                                                                       
            StandardScaler(),                                                                           
            KNeighborsClassifier(
                n_neighbors=N_NEIGHBORS,
                weights=WEIGHTS,
                algorithm=ALGORITHM,
                leaf_size=LEAF_SIZE,
                p=P,
                metric=METRIC,
            )
        )
        self.clf.fit(X, y.ravel())

    def predict_proba(self, X):
        y_proba = self.clf.predict_proba(X)
        return y_proba
