import os
import time
import pickle

from copy import deepcopy

import pandas as pd

from openfe import OpenFE, tree_to_formula, transform

import rampds as rs

from rampds.openfe_utils.openfe_utils import OpenFEUtils
from rampds.openfe_utils.training import run_ramp_experiment
from rampds.openfe_utils.utils import (
    DataFramePreprocessor,
    save_ramp_setup_kit_data,
    get_new_columns_name_dtype_and_check,
    generate_new_feature_types,
    extract_metadata_infos,
)


class OpenFEFeatureEngineering:
    def __init__(
        self,
        # data inputs
        train_df, 
        test_df,
        metadata,
        data_name,
        # scoring inputs
        n_cv_folds=30,
        clean_ramp_kits=False,
        # openfe parameters
        verbose=False, 
        max_new_feat_ratio=30, 
        n_jobs_gen=16, 
        min_cand_feat=10000, 
        n_data_blocks=2, 
        feat_boost=False,
        # feature selection 
        feat_selec_method="grid_search",
        n_feat_to_test=[1, 2, 5, 10, 15, 20, 30, 50, 100, 200],
        # results storing
        results_path="./",
        ramp_dirs_path=None,
    ):  
        # data inputs
        self.train_df = train_df
        self.test_df = test_df
        self.metadata = metadata
        self.data_name = data_name # TODO see if not directly take things from metadata
        
        # scorer
        self.n_cv_folds = n_cv_folds
        self.clean_ramp_kits = clean_ramp_kits

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

        self.exp_type = OpenFEUtils.get_experiment_type(
            min_cand_feat, n_data_blocks, feat_boost, feat_selec_method
        )
        self.exp_name = f"{self.data_name}_{self.exp_type}"
        # TODO: see how we handle the creation of setup kits for openfe experiments
        # Where do we put them ? Do we automatically delete them ? etc.
        self.results_dir = os.path.join(results_path, f"openfe_{self.exp_type}", self.data_name)
        # TODO: clean this
        self.ramp_dirs_path = ramp_dirs_path if ramp_dirs_path is not None else self.results_dir
        self._setup_paths()
        self._created_dirs()

        self.load_data()
        self.df_preprocessor = DataFramePreprocessor()

    # ==========================================================================
    # --- Public Methods ---
    # ==========================================================================

    @rs.actions.ramp_action
    def run_feature_engineering_and_selection(self):
        print("\nStarting OpenFE feature engineering experiment...")        
        self.start_time = time.time()
        self._print_experiment_setup()

        self.original_score = self.score_dataset(
            self.train_df, self.test_df, self.metadata, n_cv_folds=self.n_cv_folds, complete_setup_kit_name=f"{self.exp_name}_original"
        )

        # preprocess for openfe
        self.preprocess_data()

        # create new OpenFE features or load them if they already exist
        self.openfe_features = self.generate_and_save_features()
        self.openfe_feat_names = self._get_new_feature_names(self.openfe_features)
        self.n_new_features = len(self.openfe_features)

        # get the scores for different number of selected features
        self.scores_df = self.feature_selection_experiment()

        # get best feature configuration and prepare best updated dataframes and metadata
        self.best_n_selec_feat, self.best_score = self._get_best_feature_configuration()
        self.updated_train_df, self.updated_test_df, self.updated_metadata = self._update_new_best_data(self.best_n_selec_feat)

        # save results and best setup kit if required # TODO: removed best setup kit
            
        self._print_final_results()

        #TODO: see what we want to return here
        result_dict = {
            "best_n_selected_features": self.best_n_selec_feat,
            "best_score": self.best_score,
            "original_score": self.original_score,
            "scores_df": self.scores_df,
            "total_time_seconds": time.time() - self.start_time
        }

        # TODO: see if we also want to save these + automatic plots in the results dir
        # see https://rnd-gitlab-eu.huawei.com/Noahs-Ark/research_projects/feature_selection_autods/-/blob/main/feature_selection/feature_engineering/openfe.py?ref_type=heads#L553

        # only return a dict without dfs and metadata cause needs to be pickled w ramp action
        return result_dict
    
    # return the actual updated in another function to avoid pickling issues
    def load_best_updated_data(self):
        return self.updated_train_df, self.updated_test_df, self.updated_metadata
    
        
    def load_data(self):
        print("\nLoading data...\n")
        self.target_column_name, self.id_column_name, self.score_name, self.prediction_type, self.objective_direction = extract_metadata_infos(self.metadata)
        self.n_original_features = len(self.test_df.columns)
      
        print(f"Loaded data with {self.n_original_features} original features.")
        print(f"Original number of features: {self.n_original_features}")
        print(f"Target column name: {self.target_column_name}")
        print(f"ID column name: {self.id_column_name}")

        print(f"\n{'-'*50}\n")

    def preprocess_data(self):
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

    def generate_and_save_features(self):
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
    
    def score_dataset(self, train_df, test_df, metadata, complete_setup_kit_name, n_cv_folds=30):
        updated_ramp_setup_kit_path = os.path.join(self.ramp_setup_kit_path, complete_setup_kit_name)

        save_ramp_setup_kit_data(
            train_df=train_df, 
            test_df=test_df, 
            metadata=metadata,
            ramp_setup_kit=updated_ramp_setup_kit_path
        )

        mean_score_value, _ = run_ramp_experiment(
            complete_setup_kit_name=complete_setup_kit_name,
            n_cv_folds_arg=n_cv_folds,
            metadata=metadata,
            base_ramp_setup_kits_path=self.ramp_setup_kit_path,
            base_ramp_kits_path=self.ramp_kit_path,
            clean_ramp_kit=self.clean_ramp_kits,
        )

        return mean_score_value

    def feature_selection_experiment(self):
        print("\n" + "=" * 50)
        print("Starting feature selection experiments")
        print("=" * 50 + "\n")

        scores_list = []
        max_features_to_test = self.max_new_features_ratio * self.n_original_features

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

                complete_setup_kit_name = f"{self.exp_name}_OpenFE_{n_selected_features}_feat"
                mean_score_value = self.score_dataset(
                    train_df=updated_train_df, 
                    test_df=updated_test_df, 
                    metadata=updated_metadata,
                    complete_setup_kit_name=complete_setup_kit_name,
                    n_cv_folds=self.n_cv_folds,
                )
         
                # mean_score_value = self._evaluate_dataset_score(updated_train_df, updated_test_df, updated_metadata)
                print(f"Mean score value for {n_selected_features} selected features: {mean_score_value}")
                scores_list.append((n_selected_features, mean_score_value))
            except Exception as e:
                print(f"Error selecting {n_selected_features} features: {e}")
                continue

        scores_df = self._create_scores_df(scores_list=scores_list)

        return scores_df
    
    # TODO: can use this to add in the 
    # def save_results(self):
    #     print("\nSaving results...")
    #     self.experiment_metadata = {
    #         "min_candidate_features": self.min_candidate_features,
    #         "n_data_blocks": self.n_data_blocks,
    #         "feature_boosting": self.feature_boosting,
    #         "feature_selection_method": self.feature_selection_method,
    #         "n_new_features": self.n_new_features,
    #         "total_time_seconds": time.time() - self.start_time,
    #         "score_name": self.score_name,
    #         "objective_direction": self.objective_direction,
    #         "data_name": self.data_name,
    #         "n_feat_to_test": self.n_feat_to_test,
    #         "original_score": self.original_score,
    #     }

    #     FileUtils.save_json(self.experiment_metadata, self.experiment_metadata_path)
    #     FileUtils.save_csv(self.scores_df, self.scores_saving_path)

    # ==========================================================================
    # --- Private Methods ---
    # ==========================================================================
    
    # TODO: potentially created directories for all these
    def _setup_paths(self):
        self.scores_saving_path = os.path.join(self.results_dir, "scores.csv")
        self.experiment_metadata_path = os.path.join(self.results_dir, "experiment_metadata.json")
        self.new_features_saving_path = os.path.join(self.results_dir, "openfe_features.pkl")
        self.ramp_setup_kit_path = os.path.join(self.ramp_dirs_path, 'ramp_setup_kits')
        self.ramp_kit_path = os.path.join(self.ramp_dirs_path, 'ramp_kits')
        self.tmp_path = os.path.join(self.results_dir, 'tmp')

    def _created_dirs(self):
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.ramp_setup_kit_path, exist_ok=True)
        os.makedirs(self.ramp_kit_path, exist_ok=True)
        os.makedirs(self.tmp_path, exist_ok=True)
    
    # Utils for getting and renaming OpenFE features 

    def _get_new_feature_names(self, features):
        original_feature_names = [tree_to_formula(f) for f in features]
        feature_names = OpenFEUtils.rename_OpenFE_columns(original_feature_names) 
        
        return feature_names

    def _rename_openfe_columns(self, selected_features, new_train_x, new_test_x):
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
    
    # Utils to update dataframes and metadata with new features
    
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

        transform_tmp_path = os.path.join(self.tmp_path, f"openfe_tmp_data_{self.exp_name}.feather")
        # TODO: used updated version of openfe to be able to specify the tmp path (custom modification)
        # transform the datasets with new OpenFE features and restore original col names
        new_train_x, new_test_x = transform(
            train_df_sanitized, 
            test_df_sanitized, 
            selected_features, 
            n_jobs=4,
            tmp_path=transform_tmp_path
        ) 

        # rename the new features with simplified names
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
    
    # Utils for handling scores and best features configuration
    
    def _create_scores_df(self, scores_list):
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

    # Utils for printing experiment setup and final results

    def _print_experiment_setup(self):
        print("\n- Experiment setup:")

        print(f"\nExperiment name: {self.exp_name}")
        print(f"Data name: {self.data_name}")
        print(f"Results directory: {self.results_dir}")
        print(f"Scores results saving path: {self.scores_saving_path}")
        print(f"Number of CV folds: {self.n_cv_folds}")
        
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
