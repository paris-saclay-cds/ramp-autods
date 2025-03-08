import numpy as np
import pandas as pd
from typing import Tuple
import rampds as rs


class DataPreprocessor(rs.BaseDataPreprocessor):
    """Drop ID column"""

    def preprocess(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
        metadata: dict,
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, dict]:
        X_train = X_train.drop(columns=[metadata["id_col"]])
        X_test = X_test.drop(columns=[metadata["id_col"]])
        metadata["data_description"]["feature_types"].pop(metadata["id_col"])
        return X_train, y_train, X_test, metadata
