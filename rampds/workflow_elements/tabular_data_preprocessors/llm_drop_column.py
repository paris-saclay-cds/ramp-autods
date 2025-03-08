import os
from copy import deepcopy
from pathlib import Path
from typing import Optional, Tuple
import json
import numpy as np
import pandas as pd
import rampds as rs
from ramphy import Hyperparameter

llm = Hyperparameter(dtype="str", default="fschat/llama-3-8B-Instruct", values=["fschat/llama-3-8B-Instruct"])

LLM = str(llm)


class DataPreprocessor(rs.BaseDataPreprocessor):
    def preprocess(
        self, X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame, metadata: dict
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, dict]:
        if "PANGU_PATH" not in os.environ:
            raise ValueError("You should set the PANGU_PATH environment variable.")

        ramp_kit_dir = Path(__file__).parent.parent
        llm_output_path = ramp_kit_dir / "llm_output"

        if not (llm_output_path / "dropped_features.json").exists():
            rs.pangu_actions.llm_drop_feature(
                pangu_root=os.environ["PANGU_PATH"], output_path=llm_output_path, kit_path=ramp_kit_dir, llm=LLM
            )
        else:
            print(
                f"We already have a suggestion of dropped features at {(llm_output_path / 'dropped_features.json')}. \
                Not asking the LLM again. If you want new ones, remove the file."
            )

        with open(llm_output_path / "dropped_features.json", "r") as f:
            self.drop_feats = json.load(f)["features_to_drop"]

        X_train = X_train.drop(self.drop_feats, axis=1)
        X_test = X_test.drop(self.drop_feats, axis=1)

        metadata = deepcopy(metadata)  # TODO purge metdata
        metadata["data_description"]["feature_types"].pop(self.drop_feats)
        return X_train, y_train, X_test, metadata
