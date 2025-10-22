import os
import time
import shutil
import pickle

from copy import deepcopy
from pathlib import Path

import pandas as pd

from openfe import OpenFE, tree_to_formula, transform

from rampds.feat_eng.openfe_utils import OpenFEUtils
from rampds.feat_eng.training import run_ramp_experiment
from rampds.feat_eng.utils import (
    DataFramePreprocessor,
    save_ramp_setup_kit_data,
    get_new_columns_name_dtype_and_check,
    generate_new_feature_types,
    extract_metadata_infos,
    FileUtils
)


# Current API because relies a lot on internal attributes of the class (self.attr)
# So a lot of functions have unclear arguments as they are not passed as input but taken from self 
class OpenFEFeatureEngineering:
    def __init__(
        self,
        # data inputs
        train_df, 
        test_df,
        metadata,
        data_name,
        # scoring inputs
        n_cv_folds=15,
        clean_ramp_kits=True,
        blend=True,
        base_predictors=["lgbm"],
        # openfe parameters
        verbose=False, 
        n_jobs_gen=16, 
        min_cand_feat=10000, 
        n_data_blocks=2, 
        feat_boost=False,
        # feature selection 
        feat_selec_method="grid_search",
        n_feat_to_test=[1, 5, 10, 15, 20, 35, 50, 100, 200, 350, 500],
        max_new_feat_ratio=7,
        # results storing
        results_path="openfe_experiments/",
        exp_version="test",
        overwrite_results_dir=True,
        **kwargs
    ):  
        # data inputs
        self.train_df = train_df
        self.test_df = test_df
        self.metadata = metadata
        self.data_name = data_name
        
        # scorer
        self.n_cv_folds = n_cv_folds
        self.clean_ramp_kits = clean_ramp_kits
        self.blend = blend
        self.base_predictors = base_predictors

        # openfe parameters
        self.max_new_features_ratio = max_new_feat_ratio
        self.n_jobs_gen = n_jobs_gen
        self.min_candidate_features = min_cand_feat
        self.n_data_blocks = n_data_blocks
        self.feature_boosting = feat_boost
        self.verbose = verbose

        # feature selection parameters
        self.feature_selection_method = feat_selec_method
        self.n_feat_to_test = n_feat_to_test

        # results storing
        self.exp_type = OpenFEUtils.get_experiment_type(
            min_cand_feat, n_data_blocks, feat_boost, feat_selec_method
        )
        self.exp_version = exp_version
        self.model_text = "blend_lgbm" if self.blend else "single_lgbm"
        self.exp_name = f"{self.data_name}_{self.exp_type}_{self.model_text}_v{self.exp_version}"
        self.results_dir = os.path.join(results_path, f"openfe_{self.exp_name}", self.data_name)
        self.ramp_dirs_path = self.results_dir
        self.overwrite_results_dir = overwrite_results_dir

        # setup paths and create dirs for results / data storage
        self._setup_paths()
        self._create_dirs()

        # load data and initialize preprocessor
        self.load_data()
        self.df_preprocessor = DataFramePreprocessor()

        # fallback for unexpected kwargs
        if kwargs:
            raise ValueError(f"Unexpected config keys: {list(kwargs.keys())}")


    # ==========================================================================
    # --- Public API Methods ---
    # ==========================================================================

    # @rs.actions.ramp_action
    def run_feature_engineering_and_selection(self):
        """Run feature engineering with OpenFE and selection process on created features.
        The results are saved in the specified results directory, and the new updated datasets are 
        stored in the class under the attributes `updated_train_df`, `updated_test_df`, and `updated_metadata`. 

        Returns:
            dict: A dictionary containing the results metadata of the experiment.
        """
        print("\nStarting OpenFE feature engineering experiment...")        
        self.start_time = time.time()
        self._print_experiment_setup()

        # Clear cache directories to ensure training starts fresh
        self._clear_cache_directories()

        # preprocess for openfe
        self.preprocess_data()

        print("\nScoring original dataset...")
        self.original_score = self.score_dataset(
            self.train_df, 
            self.test_df, 
            self.metadata, 
            complete_setup_kit_name=f"{self.exp_name}_original"
        )
        print(f"Original score: {self.original_score}")

        # create new OpenFE features or load them if they already exist
        self.openfe_features = self.generate_and_save_features()
        self.openfe_feat_names = self._get_new_feature_names(self.openfe_features)
        self.n_new_features = len(self.openfe_features)

        # get the scores for different number of selected features
        self.scores_df = self.feature_selection_experiment()

        # get best feature configuration and prepare best updated dataframes and metadata
        self.best_n_selec_feat, self.best_score = self._get_best_feature_configuration()
        self.updated_train_df, self.updated_test_df, self.updated_metadata = self._update_new_best_data(self.best_n_selec_feat)

        # save results to disk and print final results
        self.save_results()
        self._print_final_results()

        # only return a dict with main results (more detailed ones are saved on disk)
        result_dict = {
            "best_n_selected_features": self.best_n_selec_feat,
            "best_score": self.best_score,
            "original_score": self.original_score,
            "scores_df": self.scores_df,
            "total_time_seconds": time.time() - self.start_time
        }

        return result_dict
    
    def load_best_updated_data(self):
        """Load the best updated dataframes and metadata.

        Returns:
            tuple: A tuple containing the updated training dataframe, updated testing dataframe, and updated metadata.
        """
        return self.updated_train_df, self.updated_test_df, self.updated_metadata
    
    
    # ==========================================================================
    # --- Feature Engineering and Selection Methods ---
    # ==========================================================================

    def generate_and_save_features(self):
        """Generate new features using OpenFE and save them in a pickle file.

        Returns:
            features: custom OpenFE feature objects generated.
        """
        print("\n" + "=" * 50)
        print("Starting feature engineering experiment with OpenFE")
        print("=" * 50 + "\n")

        tmp_save_name = f'openfe_tmp_{self.exp_name}_xx.feather'
        tmp_save_path = os.path.join(self.tmp_path, tmp_save_name)

        # generate new features with OpenFE
        ofe = OpenFE()
        features = ofe.fit(
            data=self.train_x_sanitized, 
            label=self.train_y_sanitized, 
            n_jobs=self.n_jobs_gen, 
            verbose=self.verbose,
            categorical_features=self.categorical_columns_sanitized,
            min_candidate_features=self.min_candidate_features,
            n_data_blocks=self.n_data_blocks,
            feature_boosting=self.feature_boosting,
            tmp_save_path=tmp_save_path 
        )

        # save the generated features as a pickle file
        with open(self.new_features_saving_path, "wb") as f:
            pickle.dump(features, f)

        # manually delete temporary file in case there was a crash
        if os.path.exists(tmp_save_path):
            os.remove(tmp_save_path)

        print(f"Generated {len(features)} new features.")
        return features
    

    def feature_selection_experiment(self):
        """Run feature selection experiments using the generated features.

        Returns:
            pd.DataFrame: A DataFrame containing the scores for each feature selection experiment.
        """
        print("\n" + "=" * 50)
        print("Starting feature selection experiments")
        print("=" * 50 + "\n")

        # init a scores list for all the different number of features to add to the original dataset
        scores_list = []
        # define a maximum number of features to test based on original features and ratio
        max_features_to_test = self.max_new_features_ratio * self.n_original_features

        # iterate through the list of number of features to test
        for n_selected_features in self.n_feat_to_test:
            print("\n" + "-" * 50)
            print(f"Running experiment with {n_selected_features} selected features")
            print("-" * 50 + "\n")

            if n_selected_features > min(self.n_new_features, max_features_to_test):
                print(f"Skipping {n_selected_features} selected features as it exceeds the number of new features generated: {self.n_new_features} or max features to test: {max_features_to_test}")
                continue
            try:
                # update dataframes and metadata with selected features to evaluate them
                selected_features = self.openfe_features[:n_selected_features]
                updated_train_df, updated_test_df, updated_metadata = self._update_dataframes_and_metadata(selected_features)

                # evaluate the new dataset
                complete_setup_kit_name = f"{self.exp_name}_OpenFE_{n_selected_features}_feat"
                mean_score_value = self.score_dataset(
                    train_df=updated_train_df, 
                    test_df=updated_test_df, 
                    metadata=updated_metadata,
                    complete_setup_kit_name=complete_setup_kit_name,
                )
         
                print(f"Mean score value for {n_selected_features} selected features: {mean_score_value}")
                scores_list.append((n_selected_features, mean_score_value))
            except Exception as e:
                print(f"Error selecting {n_selected_features} features: {e}")
                continue
        
        # remove the temporary ramp kits if specified
        if self.clean_ramp_kits:
            shutil.rmtree(self.ramp_setup_kit_path)
            shutil.rmtree(self.ramp_kit_path)
                
        # create a scores dataframe from the scores list
        scores_df = self._create_scores_df(scores_list=scores_list)
        return scores_df
    
    
    # ==========================================================================
    # --- Initial Data Handling Methods ---
    # =========================================================================
        
    def load_data(self):
        """Load and initialize data-related attributes
        """
        print("\nLoading data...\n")
        self.target_column_name, self.id_column_name, self.score_name, self.prediction_type, self.objective_direction = extract_metadata_infos(self.metadata)
        self.n_original_features = len(self.test_df.columns)
      
        print(f"Loaded data with {self.n_original_features} original features.")
        print(f"Original number of features: {self.n_original_features}")
        print(f"Target column name: {self.target_column_name}")
        print(f"ID column name: {self.id_column_name}")

        print(f"\n{'-'*50}\n")

    def preprocess_data(self):
        """Preprocess the training data for OpenFE feature generation.
        """
        print("\nPreprocessing data...")
        self.df_preprocessor = DataFramePreprocessor()
        
        # obtain x and y datasets from train_df, and sanitize their names
        print(f"\nRemoving target and ID columns from training data, creating target data...")
        self.train_x = self.train_df.drop(columns=[self.target_column_name, self.id_column_name])
        self.train_y = self.train_df[[self.target_column_name]]
        
        print(f"\nFilling missing values and sanitizing column names...")
        self.train_x_sanitized = self.df_preprocessor.auto_fill_missing_df(self.df_preprocessor.sanitize_dataframe_columns(self.train_x))
        self.train_y_sanitized = self.df_preprocessor.sanitize_dataframe_columns(self.train_y)

        # obtain the categorical columns from metadata and sanitize their names
        self.categorical_columns = [col for col, dtype in self.metadata["data_description"]["feature_types"].items() if dtype in ("cat", "text")]
        self.categorical_columns_sanitized = [self.df_preprocessor.sanitize_name(c) for c in self.categorical_columns]
        print(f"Categorical columns: {self.categorical_columns}")

        print(f"\n{'-'*50}\n")


    # ==========================================================================
    # --- Scoring Methods ---
    # =========================================================================
    
    def score_dataset(
            self, 
            train_df, 
            test_df, 
            metadata, 
            complete_setup_kit_name, 
        ):
        """ Score the dataset using ramp experiment. 
        Save the updated ramp directories if self.clean_ramp_kits is True.
        Use a blend of models for evaluation if self.blend is True.

        Args:
            train_df (pd.DataFrame): The training dataframe.
            test_df (pd.DataFrame): The testing dataframe.
            metadata (dict): The metadata dictionary.
            complete_setup_kit_name (str): The name of the complete setup kit.

        Returns:
            float: The mean score value obtained from the ramp experiment.
        """
        updated_ramp_setup_kit_path = os.path.join(self.ramp_setup_kit_path, complete_setup_kit_name)

        save_ramp_setup_kit_data(
            train_df=train_df, 
            test_df=test_df, 
            metadata=metadata,
            ramp_setup_kit=updated_ramp_setup_kit_path
        )

        mean_score_value, _ = run_ramp_experiment(
            complete_setup_kit_name=complete_setup_kit_name,
            n_cv_folds_arg=self.n_cv_folds,
            prediction_type=self.prediction_type,
            base_ramp_setup_kits_path=self.ramp_setup_kit_path,
            base_ramp_kits_path=self.ramp_kit_path,
            clean_ramp_kit=self.clean_ramp_kits,
            blend=self.blend,
            base_predictors=self.base_predictors,
        )

        return mean_score_value


    # ==========================================================================
    # --- Setup Methods ---
    # ==========================================================================
    
    def _setup_paths(self):
        """ Setup paths for results and data storage.
        """
        self.scores_saving_path = os.path.join(self.results_dir, "scores.csv")
        self.experiment_metadata_path = os.path.join(self.results_dir, "experiment_metadata.json")
        self.new_features_saving_path = os.path.join(self.results_dir, "openfe_features.pkl")
        self.ramp_setup_kit_path = os.path.join(self.ramp_dirs_path, 'ramp_setup_kits')
        self.ramp_kit_path = os.path.join(self.ramp_dirs_path, 'ramp_kits')
        self.tmp_path = os.path.join(self.results_dir, 'tmp')


    def _create_dirs(self):
        """ Create necessary directories for results and data storage.

        Raises:
            FileExistsError: If the results directory already exists and self.overwrite_results_dir is set to False.
        """
        if os.path.exists(self.results_dir) and os.path.exists(self.scores_saving_path):
            print(f"Warning: results directory {self.results_dir} already exists. Contents may be overwritten.")
            if not self.overwrite_results_dir:
                raise FileExistsError(f"Results directory {self.results_dir} already exists and overwrite_results_dir is set to False.")
            else:
                print(f"Overwriting contents of {self.results_dir}...")
                shutil.rmtree(self.results_dir)

        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.ramp_setup_kit_path, exist_ok=True)
        os.makedirs(self.ramp_kit_path, exist_ok=True)
        os.makedirs(self.tmp_path, exist_ok=True)


    def _clear_cache_directories(self):
        """Clear cache directories for clean experiment runs."""
        cache_dir = Path("cache")
        print(f"Checking cache directory at {cache_dir.resolve()}")
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            print(f"Cleared cache directory at {cache_dir.resolve()}")
        else:
            print("No cache directory found.")

        catboost_info_dir = Path("catboost_info")
        print(f"Checking catboost_info directory at {catboost_info_dir.resolve()}")
        if catboost_info_dir.exists():
            shutil.rmtree(catboost_info_dir)
            print(f"Cleared catboost_info directory at {catboost_info_dir.resolve()}")
        else:
            print("No catboost_info directory found.")


    # ==========================================================================
    # --- PRIVATE FEATURE HANDLING METHODS ---
    # ==========================================================================

    def _get_new_feature_names(self, features):
        """ Get informative names for the new OpenFE features.
        e.g 'autoFE_f_0' --> 'feature1__div__feature2'

        Args:
            features (list): List of feature names.

        Returns:
            list: List of informative feature names.
        """
        original_feature_names = [tree_to_formula(f) for f in features]
        feature_names = OpenFEUtils.rename_OpenFE_columns(original_feature_names) 
        
        return feature_names


    def _rename_openfe_columns(self, selected_features, new_train_x, new_test_x):
        """ Rename the columns of the new OpenFE features in the transformed datasets.

        Args:
            selected_features (list): List of selected feature names.
            new_train_x (pd.DataFrame): Transformed training dataset.
            new_test_x (pd.DataFrame): Transformed testing dataset.

        Returns:
            tuple: Updated training and testing datasets with renamed columns.
        """
        # Generate the default OpenFE names ('autoFE_f_0', 'autoFE_f_1', ...)
        default_openfe_names = [f'autoFE_f_{i}' for i in range(len(selected_features))]
        
        # Generate the desired simplified names
        correct_feature_names = self._get_new_feature_names(selected_features)
        
        # Create a mapping from default names to simplified names
        rename_map = dict(zip(default_openfe_names, correct_feature_names))

        # Rename the columns in the new dataframes
        new_train_x.rename(columns=rename_map, inplace=True)
        new_test_x.rename(columns=rename_map, inplace=True)

        return new_train_x, new_test_x
    

    def _update_dataframes_and_metadata(self, selected_features):
        """ Updates the train and test DataFrames and metadata with the selected features.

        Args:
            selected_features (list): List of selected feature names to add.
        
        Returns:
            tuple: Updated train and test DataFrames, and updated metadata.
        """
        updated_train_df, updated_test_df = self._add_features_to_dataframes(selected_features)
        updated_metadata = self._add_features_to_metadata(updated_train_df, updated_test_df)
        return updated_train_df, updated_test_df, updated_metadata


    def _add_features_to_dataframes(self, selected_features):
        """ Adds new features to the original train and test DataFrames.

        Args:
            selected_features (list): List of selected feature names to add.

        Returns:
            tuple: Updated train and test DataFrames with new features added.
        """
        train_df_sanitized = self.df_preprocessor.sanitize_dataframe_columns(self.train_df)
        test_df_sanitized = self.df_preprocessor.sanitize_dataframe_columns(self.test_df)

        # need to use custom version of openfe to be able to specify the tmp path (custom modification compared to original repo)
        transform_tmp_path = os.path.join(self.tmp_path, f"openfe_tmp_data_{self.exp_name}.feather")

        # transform the datasets with new OpenFE features and restore original col names
        new_train_x, new_test_x = transform(
            train_df_sanitized, 
            test_df_sanitized, 
            selected_features, 
            n_jobs=4,
            tmp_path=transform_tmp_path
        ) 

        # rename the new features with relevant openfe names
        new_train_x, new_test_x = self._rename_openfe_columns(
            selected_features,
            new_train_x,
            new_test_x
        )

        # manually delete temporary file in case there was a crash
        if os.path.exists(transform_tmp_path):
            # safe_delete(transform_tmp_path)
            os.remove(transform_tmp_path)

        # revert the sanitized column names to original ones
        updated_train_df = self.df_preprocessor.revert_column_names(new_train_x)
        updated_test_df = self.df_preprocessor.revert_column_names(new_test_x)

        # Reorder columns to put target at last
        cols = updated_train_df.columns.tolist()
        cols.remove(self.target_column_name)
        updated_train_df = updated_train_df[cols + [self.target_column_name]]

        return updated_train_df, updated_test_df
    

    def _add_features_to_metadata(self, updated_train_df, updated_test_df):
        """ Adds new feature types to the metadata based on the updated DataFrames.

        Args:
            updated_train_df (DataFrame): updated train DataFrame with new features.
            updated_test_df (DataFrame): updated test DataFrame with new features.
        """
        # get the new column names and types and the new feature types for metadata
        new_column_names, new_column_types = get_new_columns_name_dtype_and_check(self.train_df, self.test_df, updated_train_df, updated_test_df)
        new_feature_types_metadata = generate_new_feature_types(new_column_names, new_column_types)

        # update metadata
        updated_metadata = deepcopy(self.metadata)
        updated_metadata["data_description"]["feature_types"].update(new_feature_types_metadata)

        print(f"\nNew columns in updated DataFrames: {new_column_names}")
        print(f"\nTypes of new columns in updated DataFrames:\n{new_column_types}")
        print(f"\nTypes of new columns for metadata: {new_feature_types_metadata}")

        return updated_metadata
    

    # ==========================================================================
    # --- PRIVATE ANALYSIS AND RESULTS METHODS ---
    # ==========================================================================
        
    def save_results(self):
        """Save the results of the feature selection experiments:
        - experiment metadata
        - scores dataframe
        - results plot
        """
        print("\nSaving results...")
        self.experiment_metadata = {
            "min_candidate_features": self.min_candidate_features,
            "n_data_blocks": self.n_data_blocks,
            "feature_boosting": self.feature_boosting,
            "feature_selection_method": self.feature_selection_method,
            "n_new_features": self.n_new_features,
            "best_n_selected_features": int(self.best_n_selec_feat),
            "best_score": self.best_score,
            "total_time_seconds": time.time() - self.start_time,
            "score_name": self.score_name,
            "objective_direction": self.objective_direction,
            "data_name": self.data_name,
            "n_feat_to_test": self.n_feat_to_test,
            "n_cv_folds": self.n_cv_folds,
            "original_score": self.original_score,
            "blend": self.blend,
        }

        # save metadata and scores df as json and csv
        FileUtils.save_json(self.experiment_metadata, self.experiment_metadata_path)
        FileUtils.save_csv(self.scores_df, self.scores_saving_path)

        experiment_label = f"blend={self.blend}, cv_folds={self.n_cv_folds}"

        # save results plot 
        OpenFEUtils.plot_and_save_scores(
            n_feat=self.scores_df["n_selected_features"].tolist(),
            scores=self.scores_df["mean_score"].tolist(),
            original_score=self.original_score,
            data_name=self.data_name,
            objective_direction=self.objective_direction,
            score_name=self.score_name,
            results_dir=self.results_dir,
            experiment_label=experiment_label,
            best_n_selected_features=int(self.best_n_selec_feat),
        )


    def _create_scores_df(self, scores_list):
        """ Create a DataFrame from the scores list.

        Args:
            scores_list (list): List of scores.

        Returns:
            pd.DataFrame: DataFrame containing the scores.
        """
        scores_df = pd.DataFrame(scores_list, columns=["n_selected_features", "mean_score"])
        scores_df["original_score"] = self.original_score
        return scores_df


    def _get_best_feature_configuration(self):
        """
        Determines the best number of features and score from the experiment results.
        Returns 0 features if no improvement over the original score is found.
        """
        print("\n" + "=" * 50)
        print("Getting best feature configuration...")
        print("=" * 50 + "\n")

        if not hasattr(self, 'scores_df') or self.scores_df is None or self.scores_df.empty:
            print("Warning: scores_df is empty. Cannot determine best feature configuration.")
            return 0, self.original_score

        print("Scores DataFrame:")
        print(self.scores_df)

        best_n, best_score = OpenFEUtils.get_best_n_selected_features(
            results_df=self.scores_df, 
            objective_direction=self.objective_direction,
            original_score=self.original_score
        )

        return best_n, best_score
    

    def _update_new_best_data(self, best_n_features):
        """
        Prepares the dataframes and metadata for the best setup kit.
        If best_n_features is 0, it returns the original data. Otherwise, it returns
        data updated with the best new features.
        """
        if best_n_features > 0:
            print(f"Improvement found. Adding the best {best_n_features} features.")
            best_features = self.openfe_features[:best_n_features]
            updated_train_df, updated_test_df, updated_metadata = self._update_dataframes_and_metadata(best_features)
        else:
            print("No improvement over original score. Using original data without new features.")
            updated_train_df, updated_test_df, updated_metadata = self.train_df, self.test_df, self.metadata
        
        return updated_train_df, updated_test_df, updated_metadata


    # ==========================================================================
    # --- DISPLAY METHODS ---
    # ==========================================================================

    def _print_experiment_setup(self):
        print("\n- Experiment setup:")

        print(f"\nExperiment name: {self.exp_name}")
        print(f"Data name: {self.data_name}")
        print(f"Results directory: {self.results_dir}")
        print(f"Scores results saving path: {self.scores_saving_path}")
        print(f"Number of CV folds: {self.n_cv_folds}")
        print(f"Blend of models for scoring: {self.blend}")
        print(f"Original number of features: {self.n_original_features}")
        print(f"Clean ramp kits after scoring: {self.clean_ramp_kits}")
        
        print(f"\nOpenFE hyperparameters:")
        print(f"Minimum candidate features: {self.min_candidate_features}")
        print(f"Number of data blocks: {self.n_data_blocks}")
        print(f"Feature boosting enabled: {self.feature_boosting}")
        print(f"Verbose mode: {self.verbose}")

        print(f"\n{'-'*50}\n")


    def _print_final_results(self):
        print("\n" + "=" * 50)
        print("Final Results:")
        print("=" * 50 + "\n")

        # general infos
        print(f"Experiment name: {self.exp_name}")
        print(f"Data name: {self.data_name}")
        print(f"Scores saving path: {self.scores_saving_path}")
        # print(f"Experiment metadata: {FileUtils.load_json(self.experiment_metadata_path)}")
        
        # detailed results
        print(f"\nBest score: {self.best_score}")
        print(f"\nScores DataFrame: {self.scores_df}")
        # print(f"Scores DataFrame:\n{FileUtils.load_csv(self.scores_saving_path)}")

        print(f"\nNumber of new features generated: {self.n_new_features}")
        print(f"Best n_selected_features: {self.best_n_selec_feat}")
        print("Best generated features:")
        for feat_name in self.openfe_feat_names[:self.best_n_selec_feat]:
            print(f" - {feat_name}")

        print("\n" + "=" * 50)
        print("Experiment completed successfully!")
