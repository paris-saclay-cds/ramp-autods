import os
import random

import numpy as np

import rampds
import rampds as rs
import rampwf as rw

from rampds.fe_utils.utils import FileUtils
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
    blend=False, # TODO: do a kinda blend config maybe 
):
    """
    Runs a RAMP experiment for a given setup kit, updates hyperparameters,
    trains the model, and returns scores. Cleans up the created RAMP kit.

    Args:
        data_name_arg (str): Base name of the dataset (e.g., "abalone").
        complete_setup_kit_name (str): Name of the complete setup kit to be used.
        n_cv_folds_arg (int): Number of cross-validation folds.
        metadata (dict): Metadata containing prediction type and other relevant information.
        seed_arg (int): Random seed.
        model_used_arg (str): Model identifier (e.g., "lgbm").
        submission_arg (str, optional): RAMP submission name. Defaults to "starting_kit".
        version_arg (str, optional): Version for RAMP kit. Defaults to "eval".
        number_arg (int, optional): Number for RAMP kit. Defaults to 1.
        clean_ramp_kit (bool, optional): Flag to clean the RAMP kit after training. Defaults to True.
        base_ramp_setup_kits_path (str, optional): Base path for RAMP setup kits. Defaults to current directory.
        base_ramp_kits_path (str, optional): Base path for RAMP kits. Defaults to current directory.

    Returns:
        dict: Contains complete_data_name, mean_score, all_scores 
    """
    ramp_kit_dir_local_actual = initialize_experiment(
        complete_setup_kit_name,
        version_arg,
        number_arg,
        seed_arg,
        base_ramp_setup_kits_path,
        base_ramp_kits_path
    )

    folds_idx = range(n_cv_folds_arg)

    # Maybe ix this hardcoded path later: lgbm.csv in rampds/fe_utils dir (e.g maybe put it in a data/ folder at base of directory)
    # Use the __file__ attribute to get the directory as a string
    base_foundation_predictors_dir = os.path.dirname(os.path.abspath(rampds.fe_utils.__file__))

    # TODO: change this name that is just refering to lgbm (can also include catboost and xgboost infos in fact)
    if blend:
        base_foundation_predictors_dir = os.path.join(base_foundation_predictors_dir, "blended_fixed_lgbm_hps")
    else:
        base_foundation_predictors_dir = os.path.join(base_foundation_predictors_dir, "fixed_lgbm_hps")
                                                
    if "regression" in prediction_type:
        foundation_predictors_dir = os.path.join(base_foundation_predictors_dir, "regressor")
    elif "classification" in prediction_type:
        foundation_predictors_dir = os.path.join(base_foundation_predictors_dir, "classifier")
    else:
        raise ValueError(f"Invalid prediction type: {prediction_type}. Must be 'regression' or 'classification'.")
    
    # train the model using fixed lgbm hps located in rampds/fe_utils/fixed_lgbm_hps or rampds/fe_utils/blended_fixed_lgbm_hps
    # this function doesn't return anything but stores the results (of single models and of the blend) on the disk, we later read them
    foundation_models(
        ramp_kit=complete_setup_kit_name,
        kit_root=base_ramp_kits_path,
        version=version_arg,
        number=number_arg,
        n_folds_hyperopt=n_cv_folds_arg,
        n_folds_final_blend=n_cv_folds_arg,
        # base_predictors=["lgbm"],
        base_predictors=["lgbm", "catboost", "xgboost"],
        # base_predictors=["lgbm", "catboost"],
        deterministic_hash=True, # this way we know the name of the trained models (lgbm_hyperopt_openfe_{i} for each model i in the blend)
        foundation_predictors_dir=foundation_predictors_dir
    )

    # if use blend look at bagged then blend score in training output
    if blend:
        score_path = os.path.join(ramp_kit_dir_local_actual, "submissions", "training_output", "bagged_then_blended_scores.csv")
        score_df = FileUtils.load_csv(score_path)
        # only look at the valid score because test one is not informative
        score = score_df.iloc[-1]["valid"]
        scores_dict = {}
    # else if use single model look at its mean score on cv folds
    else:
        # if only one model is trained with deterministic hash we know it will be called lgbm_hyperopt_openfe_0
        trained_submission = f"lgbm_hyperopt_openfe_0"
        # retrieve the score with rampds scoring functions
        problem = rw.utils.assert_read_problem(ramp_kit_dir_local_actual)
        score = _mean_score(
                trained_submission, folds_idx, problem.score_types[0], ramp_kit_dir_local_actual
        )
        scores_dict = {}
        scores_dict["mean_score"] = score


    cleanup_ramp_kit(ramp_kit_dir_local_actual, clean_ramp_kit)

    return score, scores_dict

from rampds.fe_utils.utils import cleanup_ramp_kit

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
    # import ramp ds as as 
    np.random.seed(seed_arg)
    random.seed(seed_arg)
    print(f"\nSetting up RAMP kit: {complete_setup_kit_name} in {base_ramp_kits_path}")
    rs.scripts.setup.setup(
        ramp_kit=complete_setup_kit_name,
        setup_root=base_ramp_setup_kits_path,
        kit_root=base_ramp_kits_path,
        version=version_arg,
        number=number_arg,
    )
    created_ramp_kit_name = f"{complete_setup_kit_name}_v{version_arg}_n{number_arg}"
    ramp_kit_dir_local_actual = os.path.join(base_ramp_kits_path, created_ramp_kit_name)
    print(f"RAMP kit created at: {ramp_kit_dir_local_actual}")
    return ramp_kit_dir_local_actual



