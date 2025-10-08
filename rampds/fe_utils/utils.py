import os
import re
import pickle
import shutil
import json

from copy import deepcopy

import numpy as np
import pandas as pd


#TODO: create a dataclass with these files paths
# data file names for paths
TRAIN = "train.csv"
TEST = "test.csv"
METADATA = "metadata.json"
PRIV_LDBD_SCORES = "private_leaderboard_scores.npy"
PUB_LDBD_SCORES = "public_leaderboard_scores.npy"
SAMPLE_SUBMISSION = "sample_submission.csv"
HPS_TEXT = 'hps_text.py'
OPENFE_TEST_DIR = "openfe_test"


class FileUtils:
    @staticmethod
    def load_json(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)

    @staticmethod
    def save_json(data, file_path):
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)

    @staticmethod
    def load_csv(file_path):
        return pd.read_csv(file_path)

    @staticmethod
    def save_csv(data, file_path):
        data.to_csv(file_path, index=False)

    @staticmethod
    def load_pickle(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
        
    @staticmethod
    def save_pickle(data, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

# df preprocessor
class DataFramePreprocessor:
    """
    A modular class for preprocessing pandas DataFrames, including sanitizing column names,
    handling missing values, and creating mappings between original and sanitized column names
    for multiple datasets using automatic hashing.
    """
    def __init__(self):
        self.column_mappings = {}

    @staticmethod
    def sanitize_name(name):
        """ Sanitizes a column name by replacing spaces and special characters with underscores,

        Args:
            name (str): The column name to sanitize.

        Returns:
            str: The sanitized column name.
        """
        # Replace spaces and special characters with underscores
        sanitized_name = name.replace(" ", "_").replace("-", "_").replace(".", "_").replace("/", "_")
        # Remove unsupported characters (e.g., `[`, `]`, `{`, `}`, etc.)
        sanitized_name = re.sub(r"[^\w_]", "", sanitized_name)
        return sanitized_name

    def sanitize_dataframe_columns(self, df):
        """ Sanitizes the column names of a DataFrame by replacing spaces and special characters with underscores,

        Args:
            df (pd.DataFrame): The DataFrame to sanitize.

        Returns:
            pd.DataFrame: The DataFrame with sanitized column names.
        """
        for col in df.columns:
            sanitized_col = self.sanitize_name(col)
            self.column_mappings[col] = sanitized_col

        sanitized_df = df.rename(columns=self.column_mappings)

        return sanitized_df

    def revert_column_names(self, df, accept_new_cols=True, verbose=False):
        """ Reverts the column names of a DataFrame to their original names based on the stored mappings.

        Args:
            df (pd.DataFrame): The DataFrame with sanitized column names.
            accept_new_cols (bool, optional): Whether to accept new columns not found in the mapping. Defaults to True.
            verbose (bool, optional): Whether to print renaming actions. Defaults to False.

        Raises:
            ValueError: If accept_new_cols is False and a column is not found in the mapping.

        Returns:
            pd.DataFrame: The DataFrame with reverted column names.
        """
        df = df.copy()

        self.reverse_mapping = {v: k for k, v in self.column_mappings.items()}

        for col in df.columns:
            if col in self.reverse_mapping:
                df.rename(columns={col: self.reverse_mapping[col]}, inplace=True)
                if verbose:
                    print(f"Renamed column {col} to {self.reverse_mapping[col]}.")
            else:
                if not accept_new_cols:
                    raise ValueError(f"Column {col} not found in mapping. Cannot revert names.")
                if verbose:
                    print(f"Column {col} not found in mapping. Skipping renaming.")
        return df

    @staticmethod
    def auto_fill_missing_col(df, col):
        """ Automatically fills missing values in a specific column of a DataFrame based on its data type.

        Args:
            df (pd.DataFrame): The DataFrame containing the column.
            col (str): The column name to fill missing values for.

        Returns:
            pd.DataFrame: The DataFrame with missing values filled in the specified column.
        """
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            df[col].fillna(series.median(), inplace=True)
        elif pd.api.types.is_categorical_dtype(series) or series.dtype == object:
            mode = series.mode()
            fill_value = mode[0] if not mode.empty else "missing"
            df[col].fillna(fill_value, inplace=True)
        elif pd.api.types.is_bool_dtype(series):
            mode = series.mode()
            fill_value = mode[0] if not mode.empty else False
            df[col].fillna(fill_value, inplace=True)
        elif np.issubdtype(series.dtype, np.datetime64):
            df[col].fillna(series.median(), inplace=True)
        else:
            df[col].fillna("missing", inplace=True)
        return df

    @staticmethod
    def auto_fill_missing_df(df):
        """ Automatically fills missing values in all columns of a DataFrame based on their data types.

        Args:
            df (pd.DataFrame): The DataFrame to fill missing values for.

        Returns:
            pd.DataFrame: The DataFrame with missing values filled in all columns.
        """
        df = df.copy()
        for col in df.columns:
            df = DataFramePreprocessor.auto_fill_missing_col(df, col)
        return df
    
    @staticmethod
    def compare_dataframes(df1, df2, columns_to_check=None, raise_error=True):
        """
        Compare two DataFrames and report differences, handling NaN values properly.
        
        Parameters:
        -----------
        df1, df2 : pandas.DataFrame
            DataFrames to compare
        columns_to_check : list, optional
            List of columns to check, if None all common columns are checked
        
        Returns:
        --------
        bool
            True if DataFrames match (considering NaN == NaN), False otherwise
        """
        if columns_to_check is None:
            columns_to_check = list(set(df1.columns) & set(df2.columns))
        
        all_match = True
        # print(f"Checking if DataFrames are identical for {len(columns_to_check)} columns:")
        
        for col in columns_to_check:
            # Handle NaN values properly by using pandas equals which treats NaN == NaN
            if df1[col].equals(df2[col]):
                # print(f"Column '{col}' matches between DataFrames")
                pass
            else:
                if raise_error:
                    raise ValueError(f"Column '{col}' differs between DataFrames")
                all_match = False
                print(f"Column '{col}' differs between DataFrames")
                
                # Find differing indices
                mask = ~(df1[col] == df2[col])
                # Also handle case where both are NaN
                mask = mask & ~(df1[col].isna() & df2[col].isna())
                
                diff_indices = mask[mask].index.tolist()
                if len(diff_indices) > 0:
                    print(f"  Found {len(diff_indices)} differences, showing first 5:")
                    for idx in diff_indices[:5]:
                        print(f"    Index {idx}: df1={df1[col].loc[idx]}, df2={df2[col].loc[idx]}")
        
        return all_match
    

def sanitize_directory_path(path):
    """
    Sanitizes only the last component of a directory path while preserving the path structure.
    
    Args:
        path (str): The directory path to sanitize
        
    Returns:
        str: Path with only the last component sanitized
    """
    # Handle special cases
    if not path:
        return path
    if path == '/':
        return path
    if path in ['.', '..', './']:
        return path
    
    # Remove trailing slashes for correct processing
    path = path.rstrip('/')
    if not path:
        return '/'
    
    # Get directory and basename components
    dir_path = os.path.dirname(path)
    base_name = os.path.basename(path)
    
    # Sanitize only the basename component
    sanitized_base = DataFramePreprocessor.sanitize_name(base_name)
    
    # Rebuild the path properly
    if dir_path:
        return os.path.join(dir_path, sanitized_base)
    else:
        # For paths with no directory component
        return sanitized_base

def get_data_paths(ramp_setup_kit_name):
    """Get the paths for the data files in the RAMP setup kit.

    Args:
        ramp_setup_kit_name (str): The name of the RAMP setup kit.

    Returns:
        _type_: A tuple containing the paths for train, test, metadata, private and public leaderboard scores, and sample submission files.
    """
    train_path = os.path.join(ramp_setup_kit_name, TRAIN)
    test_path = os.path.join(ramp_setup_kit_name, TEST)
    metadata_path = os.path.join(ramp_setup_kit_name, METADATA)
    priv_ldbd_scores_path = os.path.join(ramp_setup_kit_name, PRIV_LDBD_SCORES)
    public_ldbd_scores_path = os.path.join(ramp_setup_kit_name, PUB_LDBD_SCORES)
    sample_submission_path = os.path.join(ramp_setup_kit_name, SAMPLE_SUBMISSION)

    return (
        train_path,
        test_path,
        metadata_path,
        priv_ldbd_scores_path,
        public_ldbd_scores_path,
        sample_submission_path,
    )

def save_ramp_setup_kit_data(
    train_df, test_df, metadata, ramp_setup_kit, additional_infos=None, method_name=None
):
    """Save the RAMP setup kit data to the specified directory.

    Args:
        train_df (pd.DataFrame): Train data DataFrame.
        test_df (pd.DataFrame): Test data DataFrame.
        metadata (dict): Metadata dictionary.
        ramp_setup_kit (str): Path to the RAMP setup kit directory.
        additional_infos (dict, optional): Additional information dictionary containing
            'priv_ldbd_scores', 'pub_ldbd_scores', 'sample_submission', and optionally
            'best_scores_infos'. If None, default values will be used.
        method_name (str, optional): Name of the method used. Defaults to None.
    """
    # Validate and sanitize directory path
    sanitized_ramp_setup_kit = sanitize_directory_path(ramp_setup_kit)
    if ramp_setup_kit != sanitized_ramp_setup_kit:
        raise ValueError(f"RAMP Setup kit name should be sanitized. Got ramp_setup_kit={ramp_setup_kit}")

    # Create directory if needed
    os.makedirs(ramp_setup_kit, exist_ok=True)

    # Get file paths
    (train_path, test_path, metadata_path, priv_ldbd_scores_path, 
     public_ldbd_scores_path, sample_submission_path) = get_data_paths(ramp_setup_kit)

    # Save core files
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    # Handle additional_infos with defaults
    if additional_infos is not None:
        # Validate required keys
        required_keys = ["priv_ldbd_scores", "pub_ldbd_scores", "sample_submission"]
        missing_keys = [key for key in required_keys if key not in additional_infos]
        if missing_keys:
            raise ValueError(f"Missing required keys in additional_infos: {missing_keys}")
        
        # Save additional files
        np.save(priv_ldbd_scores_path, additional_infos["priv_ldbd_scores"])
        np.save(public_ldbd_scores_path, additional_infos["pub_ldbd_scores"])
        additional_infos["sample_submission"].to_csv(sample_submission_path, index=False)
        
        # Save optional best scores info
        if "best_scores_infos" in additional_infos:
            FileUtils.save_json(additional_infos["best_scores_infos"], 
                              os.path.join(ramp_setup_kit, "best_scores_infos.json"))
    else:
        # Create default files when additional_infos is None
        np.save(priv_ldbd_scores_path, np.array([]))
        np.save(public_ldbd_scores_path, np.array([]))
        # Create empty sample submission based on test data structure
        sample_df = pd.DataFrame({"id": range(len(test_df))})
        sample_df.to_csv(sample_submission_path, index=False)

    # Handle method name file
    if method_name:
        _save_method_file(ramp_setup_kit, method_name)
    
    print(f"Saved RAMP setup kit data to {ramp_setup_kit}")

def _save_method_file(ramp_setup_kit, method_name):
    """Helper function to save method name file, removing any existing .md files."""
    # Remove existing .md files
    for file in os.listdir(ramp_setup_kit):
        if file.endswith(".md"):
            os.remove(os.path.join(ramp_setup_kit, file))
    
    # Save new method file
    method_file_path = os.path.join(ramp_setup_kit, f"{method_name}.md")
    with open(method_file_path, "w") as f:
        f.write(f"Method Name: {method_name}\n")


def load_ramp_setup_kit_data(
    kit_dir_path: str, load_additional_infos: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict | None]:
    """Load the RAMP setup kit data from the specified directory.

    Args:
        kit_dir_path (str): Path to the RAMP setup kit directory.
        load_additional_infos (bool): Whether to load additional info files.
            If False, returns None for additional_infos.

    Raises:
        FileNotFoundError: If core files (train, test, metadata) are not found.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, dict, dict | None]: A tuple containing 
            train DataFrame, test DataFrame, metadata dictionary, and optional 
            additional information dictionary.
    """
    if not os.path.exists(kit_dir_path):
        raise FileNotFoundError(f"Directory does not exist: {kit_dir_path}")
    
    # Get file paths
    (train_path, test_path, metadata_path, priv_ldbd_scores_path, 
     public_ldbd_scores_path, sample_submission_path) = get_data_paths(kit_dir_path)

    # Load core files (required)
    train_df = _load_required_file(train_path, "train", pd.read_csv)
    test_df = _load_required_file(test_path, "test", pd.read_csv)
    
    with open(metadata_path, "r") as f:
        metadata_content = json.load(f)

    # Load additional info files (optional)
    additional_infos = None
    if load_additional_infos:
        additional_infos = {
            "priv_ldbd_scores": _load_optional_file(priv_ldbd_scores_path, np.load),
            "pub_ldbd_scores": _load_optional_file(public_ldbd_scores_path, np.load),
            "sample_submission": _load_optional_file(sample_submission_path, pd.read_csv),
        }

    print(f"Loaded RAMP setup kit from {kit_dir_path} (train: {train_df.shape}, test: {test_df.shape})")
    return train_df, test_df, metadata_content, additional_infos


def _load_required_file(file_path: str, file_type: str, loader_func):
    """Helper function to load required files with proper error handling."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_type.capitalize()} file not found: {file_path}")
    return loader_func(file_path)


def _load_optional_file(file_path: str, loader_func):
    """Helper function to load optional files, returning None if not found."""
    if os.path.exists(file_path):
        return loader_func(file_path)
    return None

def get_new_columns_name_dtype_and_check(train_df, test_df, updated_train_df, updated_test_df):
    """Verify if the original info of the dataframes is maintained after updates with addition of new columns, and return the new column names.
    This function checks that the updated train and test dataframes contain the same number of rows as the original datasets,
    and that all the original columns are kept and have not been altered.

    Args:
        train_df (pd.DataFrame): Original training dataframe.
        test_df (pd.DataFrame): Original test dataframe.
        updated_train_df (pd.DataFrame): Updated training dataframe after feature generation.
        updated_test_df (pd.DataFrame): Updated test dataframe after feature generation.
    """
    # Check that the number of rows is the same
    assert len(train_df) == len(updated_train_df), "Number of rows in updated train dataframe does not match original."
    assert len(test_df) == len(updated_test_df), "Number of rows in updated test dataframe does not match original."

    new_train_columns = updated_train_df.columns.difference(train_df.columns)
    new_test_columns = updated_test_df.columns.difference(test_df.columns)
    assert set(new_train_columns) == set(new_test_columns), "New columns in train and test DataFrames do not match."
    new_column_names = list(new_train_columns)

    print(new_column_names)
    new_train_column_types = updated_train_df[new_column_names].dtypes
    # new_test_column_types = updated_test_df[new_column_names].dtypes

    # TODO: I don't remember why I commented this but I think it led to undesired errors
    # assert (new_train_column_types == new_test_column_types).all(), "New columns in train and test DataFrames have different types."
    
    new_column_types = new_train_column_types

    # Check that all original columns are present and unchanged
    for col in train_df.columns:
        assert col in updated_train_df.columns, f"Column '{col}' missing in updated train dataframe."
        assert train_df[col].equals(updated_train_df[col]), f"Column '{col}' has been altered in updated train dataframe."

    for col in test_df.columns:
        assert col in updated_test_df.columns, f"Column '{col}' missing in updated test dataframe."
        assert test_df[col].equals(updated_test_df[col]), f"Column '{col}' has been altered in updated test dataframe."

    return new_column_names, new_column_types


def generate_new_feature_types(new_column_names, new_column_types):
    """Generate new feature types for metadata based on column names and types.

    Args:
        new_column_names (list): List of new column names.
        new_column_types (pd.Series): Series of new column types.

    Returns:
        dict: A dictionary mapping new column names to their feature types.
    """
    new_feature_types = {}
    for feature, dtype in zip(new_column_names, new_column_types):
        # add either num or cat type based on the dtype of the new column

        feature_type = 'num' if dtype in ['int64', 'float64'] else 'cat' 
        print(f"Feature: {feature}, dype: {dtype} --> type: {feature_type}")
        new_feature_types[feature] = feature_type
    return new_feature_types


def extract_metadata_infos(metadata, print_value=False):
    """Extracts relevant information from the metadata dictionary.

    Args:
        metadata (dict): A dictionary containing metadata information.

    Returns:
        tuple: A tuple containing target_column, id_column, score_name, prediction_type, objective direction.
    """
    target_column_name = metadata["data_description"]["target_cols"][0]
    id_column_name = metadata["id_col"]
    score_name = metadata["score_name"]
    prediction_type = metadata['prediction_type']
    objective_direction = get_objective_direction(score_name)

    if print_value:
        print("\nExtracted metadata information:")
        print(f"- Target column name: {target_column_name}")
        print(f"- ID column name: {id_column_name}")
        print(f"- Score name: {score_name}")
        print(f"- Prediction type: {prediction_type}")
        print(f"- Objective direction: {objective_direction}")

    return target_column_name, id_column_name, score_name, prediction_type, objective_direction


def cleanup_ramp_kit(ramp_kit_dir_local_actual, clean_ramp_kit):
    """Cleans up the RAMP kit directory if specified.

    Args:
        ramp_kit_dir_local_actual (str): Path to the local RAMP kit directory.
        clean_ramp_kit (bool): Flag indicating whether to clean the RAMP kit.
    """
    if clean_ramp_kit and os.path.exists(ramp_kit_dir_local_actual):
        print(f"Cleaning up RAMP kit directory: {ramp_kit_dir_local_actual}")
        # TODO: remove safe delete and replace with shutil.rmtree (even if less safe)
        # safe_delete(ramp_kit_dir_local_actual)
        shutil.rmtree(ramp_kit_dir_local_actual)
        print(f"Deleted directory: {ramp_kit_dir_local_actual}")


def get_objective_direction(score_name: str, print_value=False) -> str:
    """Determines the objective direction based on the score name.

    Args:
        score_name (str): score name to determine the objective direction.
        print_value (bool, optional): wether to print score name / objective directiohn. Defaults to False.

    Raises:
        ValueError: If the score name is unknown.

    Returns:
        str: The objective direction corresponding to the score name.
    """
    if score_name == "accuracy":
        objective_direction = "maximize"
    elif score_name == "log_loss":
        objective_direction = "minimize"
    elif score_name == "f1":
        objective_direction = "maximize"
    elif score_name == "auc":
        objective_direction = "maximize"
    elif score_name == "average_precision":
        objective_direction = "maximize"
    elif score_name == "rmsle":
        objective_direction = "minimize"
    elif score_name == "rmse":
        objective_direction = "minimize"
    elif score_name == "r2":
        objective_direction = "maximize"
    elif score_name == "mae":
        objective_direction = "minimize"
    elif score_name == "nll":
        objective_direction = "minimize"
    elif score_name == "mcc":
        objective_direction = "maximize"
    elif score_name == "rmse":
        objective_direction = "minimize"
    elif score_name == "ngini":
        objective_direction = "maximize"
    elif score_name == "kappa":
        objective_direction = "maximize"
    elif score_name == "medae":
        objective_direction = "minimize"
    elif score_name == "f1-micro":
        objective_direction = "maximize"
    elif score_name == "mape":
        objective_direction = "minimize"

    else:
        raise ValueError(f"Unknown score name: {score_name}")
    
    if print_value:
        print(f"Score name: {score_name}")
        print(f"Objective direction: {objective_direction}")

    return objective_direction