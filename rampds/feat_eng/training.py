import os
import random

from pathlib import Path

import numpy as np

import rampds
import rampds as rs
import rampwf as rw

from rampds.feat_eng.utils import FileUtils, cleanup_ramp_kit
from rampds.scripts.foundation import foundation_models
from rampds.actions import _mean_score


def run_ramp_experiment(
    complete_setup_kit_name: str,
    n_cv_folds_arg: int,
    prediction_type: str,
    seed_arg: int = 42,
    version_arg: str = "eval",
    number_arg: int = 1,
    clean_ramp_kit: bool = True,
    base_ramp_setup_kits_path: str = ".",
    base_ramp_kits_path: str = ".",
    blend=False,
    base_predictors=["lgbm"],
):
    """
    Runs a RAMP experiment for a given setup kit.

    Args:
        complete_setup_kit_name: Name of the complete setup kit to be used
        n_cv_folds_arg: Number of cross-validation folds
        prediction_type: Type of prediction ('regression', 'classification', etc.)
        seed_arg: Random seed for reproducibility
        version_arg: Version for RAMP kit
        number_arg: Number for RAMP kit
        clean_ramp_kit: Whether to clean the RAMP kit after training
        base_ramp_setup_kits_path: Base path for RAMP setup kits
        base_ramp_kits_path: Base path for RAMP kits
        blend: Whether to use blended models
        base_predictors: List of base predictors to use

    Returns:
        dict: Contains 'score', 'scores_dict', and 'experiment_info'
    
    Raises:
        ValueError: If prediction_type is invalid
        FileNotFoundError: If required files are missing
    """
    ramp_kit_dir_local_actual = initialize_experiment(
        complete_setup_kit_name,
        version_arg,
        number_arg,
        seed_arg,
        base_ramp_setup_kits_path,
        base_ramp_kits_path
    )

    # get models hyperparameters directory
    foundation_predictors_dir = _get_foundation_predictors_dir(prediction_type, blend)
    
    # training with foundation models function
    foundation_models(
        ramp_kit=complete_setup_kit_name,
        kit_root=base_ramp_kits_path,
        version=version_arg,
        number=number_arg,
        n_folds_hyperopt=n_cv_folds_arg,
        n_folds_final_blend=n_cv_folds_arg,
        base_predictors=base_predictors,
        deterministic_hash=True, # this way we know the name of the trained models (lgbm_hyperopt_openfe_{i} for each model i in the blend)
        foundation_predictors_dir=foundation_predictors_dir
    )

    # retrieve results
    score, scores_dict = _extract_scores(ramp_kit_dir_local_actual, blend, n_cv_folds_arg, base_predictors)

    # clean up ramp kit directory if specified
    cleanup_ramp_kit(ramp_kit_dir_local_actual, clean_ramp_kit)

    return score, scores_dict


def get_prediction_type(prediction_type):
    """Return either 'regression' or 'classification' based on the prediction type.

    Args:
        prediction_type (str): str prediction type
    """
    prediction_type_map = {
        "regression": "regression",
        "classification": "classification",
        "binary_classification": "classification",
        "multiclass_classification": "classification",
        "binary classification": "classification",
        "multi-class classification": "classification",
    }

    try:
        prediction_type = prediction_type_map[prediction_type]
    except KeyError:
        raise ValueError(
            f"Invalid prediction type: {prediction_type}. "
            "Must be one of 'regression', 'binary_classification', 'multiclass_classification', or 'multi-class classification'."
        )
    return prediction_type 


def initialize_experiment(
    complete_setup_kit_name, version_arg, number_arg, seed_arg, base_ramp_setup_kits_path, base_ramp_kits_path
):
    """Initializes the RAMP experiment by setting up the RAMP kit.

    Args:
        complete_setup_kit_name (str):  Name of the complete setup kit to be used.
        version_arg (str): Version of the RAMP kit.
        number_arg (str): Number of the RAMP kit.
        seed_arg (int): Seed for random number generation.
        base_ramp_setup_kits_path (str):  Base path for RAMP setup kits.
        base_ramp_kits_path (str): Base path for RAMP kits.

    Returns:
        str: Path to the created RAMP kit directory.
    """
    # fix seeds
    np.random.seed(seed_arg)
    random.seed(seed_arg)

    # Set up the RAMP kit
    print(f"\nSetting up RAMP kit: {complete_setup_kit_name} in {base_ramp_kits_path}")
    rs.scripts.setup.setup(
        ramp_kit=complete_setup_kit_name,
        setup_root=base_ramp_setup_kits_path,
        kit_root=base_ramp_kits_path,
        version=version_arg,
        number=number_arg,
    )

    # return the path to the created ramp kit
    created_ramp_kit_name = f"{complete_setup_kit_name}_v{version_arg}_n{number_arg}"
    ramp_kit_dir_local_actual = os.path.join(base_ramp_kits_path, created_ramp_kit_name)
    print(f"RAMP kit created at: {ramp_kit_dir_local_actual}")
    return ramp_kit_dir_local_actual


def _get_foundation_predictors_dir(prediction_type: str, blend: bool) -> str:
    """Get the foundation predictors directory path."""
    hyperparameters_data_dir = Path(__file__).parent / "data"
    
    model_type_dir = "blend_models_hps" if blend else "single_model_hps"
    base_dir = os.path.join(hyperparameters_data_dir, model_type_dir)
    
    prediction_type_normalized = get_prediction_type(prediction_type)
    task_dir = "regressor" if prediction_type_normalized == "regression" else "classifier"
    
    return os.path.join(base_dir, task_dir)


def _extract_scores(ramp_kit_dir: str, blend: bool, n_cv_folds: int, base_predictors: list) -> tuple:
    """Extract scores from trained models."""
    if blend:
        return _extract_blend_scores(ramp_kit_dir)
    else:
        assert len(base_predictors) == 1, "Single model extraction requires exactly one base predictor."
        return _extract_single_model_scores(ramp_kit_dir, n_cv_folds, base_predictors)


def _extract_blend_scores(ramp_kit_dir: str) -> tuple:
    """Extract scores from blended models."""
    # if use blend look at bagged then blend score in training output
    score_path = os.path.join(ramp_kit_dir, "submissions", "training_output", "bagged_then_blended_scores.csv")
    
    if not os.path.exists(score_path):
        raise FileNotFoundError(f"Blend scores file not found: {score_path}")
    
    score_df = FileUtils.load_csv(score_path)
    # only look at the valid score because test one is not informative
    score = score_df.iloc[-1]["valid"]
    
    return score, {"blend_score": score}


def _extract_single_model_scores(ramp_kit_dir: str, n_cv_folds: int, base_predictors: list) -> tuple:
    """Extract scores from single model."""
    predictor = base_predictors[0]
    trained_submission = f"{predictor}_hyperopt_openfe_0"
    # if a lgbm is trained with deterministic hash, its trained submission dir is lgbm_hyperopt_openfe_0
    
    folds_idx = range(n_cv_folds)
    problem = rw.utils.assert_read_problem(ramp_kit_dir)
    score = _mean_score(trained_submission, folds_idx, problem.score_types[0], ramp_kit_dir)
    
    return score, {"mean_score": score}