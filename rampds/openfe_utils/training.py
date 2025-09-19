import os
import random
import numpy as np
import rampds as rs



# TODO: surely modify the API to only give necessary args
def run_ramp_experiment(
    data_name_arg: str,
    complete_setup_kit_name: str,
    n_cv_folds_arg: int,
    metadata: dict,
    seed_arg: int = 42,
    model_used_arg: str = "lgbm",
    submission_arg: str = "starting_kit",
    version_arg: str = "eval",
    number_arg: int = 1,
    clean_ramp_kit: bool = True,
    base_ramp_setup_kits_path: str = ".",
    base_ramp_kits_path: str = ".",
    use_fixed_hps: bool = False,
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
    # TODO: do this with the ramp hyperopt function
    # update_model_hyperparameters(
    #     ramp_kit_dir_local_actual, 
    #     submission_arg, 
    #     model_used_arg, 
    #     data_name_arg,
    #     metadata,
    #     use_fixed_hps
    # )
    print(f"\nStarting training for {ramp_kit_dir_local_actual}, submission {submission_arg}")
    scores_dict = rs.actions.train(
        ramp_kit_dir=ramp_kit_dir_local_actual,
        submission=submission_arg,
        fold_idxs=range(n_cv_folds_arg),
        force_retrain=True,
    )
    print(f"Training completed. Scores: {scores_dict}")
    cleanup_ramp_kit(ramp_kit_dir_local_actual, clean_ramp_kit)
    return scores_dict["mean_score"], scores_dict

from rampds.openfe_utils.utils import cleanup_ramp_kit

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




# def update_model_hyperparameters(
#     ramp_kit_dir_local_actual, 
#     submission_arg, 
#     model_used_arg, 
#     data_name_arg, 
#     metadata,
#     use_fixed_hps
# ):
#     """Updates the hyperparameters of the model in the RAMP kit based on the challenge's hyperparameter file.

#     Args:
#         ramp_kit_dir_local_actual (str):  Path to the local RAMP kit directory.
#         submission_arg (str):  Name of the RAMP submission to be updated.
#         model_used_arg (str): Model identifier (e.g., "lgbm").
#         data_name_arg (str): Base name of the dataset (e.g., "abalone").
#         metadata (dict): Metadata containing prediction type and other relevant information.
#         use_fixed_hps (bool): Flag to indicate whether to use fixed hyperparameters.

#     Raises:
#         FileNotFoundError: If the hyperparameter file is not found.
#     """
#     metadata_pred_type = metadata["prediction_type"]
#     prediction_type = get_prediction_type(metadata_pred_type)
#     model_file_name = "regressor" if prediction_type == "regression" else "classifier"
#     model_file_name_py = f"{model_file_name}.py"

#     # find the path of the new hps text
#     if use_fixed_hps:
#         new_hps_py_path = os.path.join(HPS_PATH, f"{model_used_arg}_fixed", model_file_name_py)
#     else:
#         new_hps_py_path = os.path.join(HPS_PATH, model_used_arg, data_name_arg, HPS_TEXT)
    
#     target_model_hps_path = os.path.join(
#         ramp_kit_dir_local_actual, "submissions", submission_arg, model_file_name_py
#     )
#     print(f"\nUpdating HPs for {target_model_hps_path} using {new_hps_py_path}")
#     if not os.path.exists(new_hps_py_path):
#         raise FileNotFoundError(f"Hyperparameter file not found: {new_hps_py_path}")
#     update_hyperparameters(
#         target_file_path=target_model_hps_path,
#         new_params_file_path=new_hps_py_path,
#         challenge_name=data_name_arg,
#     )
#     print("Hyperparameters updated.")



# def update_hyperparameters(
#     target_file_path,
#     new_params_file_path,
#     challenge_name,
# ):
#     """Update hyperparameters in a target file with content from a new parameters file.

#     Args:
#         target_file_path (str): Path to the target file.
#         new_params_file_path (str): Path to the new parameters file.
#         challenge_name (str): Name of the challenge.

#     Raises:
#         FileNotFoundError: If the target file is not found.
#         FileNotFoundError: If the new parameters file is not found.
#     """
#     START_MARKER = "# RAMP START HYPERPARAMETERS"
#     END_MARKER = "# RAMP END HYPERPARAMETERS"

#     # Validate input file paths
#     if not os.path.exists(target_file_path):
#         raise FileNotFoundError(f"Target file '{target_file_path}' not found.")
#     if not os.path.exists(new_params_file_path):
#         raise FileNotFoundError(f"New parameters file '{new_params_file_path}' not found.")

#     # 1. Read the new parameters content
#     try:
#         with open(new_params_file_path, "r") as f:
#             # Ensure new_params_content ends with a newline if not empty
#             new_params_content = f.read()
#             if new_params_content and not new_params_content.endswith("\n"):
#                 new_params_content += "\n"
#     except IOError as e:
#         print(f"Error reading new parameters file '{new_params_file_path}': {e}")
#         return

#     # 2. Read the target file content
#     try:
#         with open(target_file_path, "r") as f:
#             target_lines = f.readlines()
#     except IOError as e:
#         print(f"Error reading target file '{target_file_path}': {e}")
#         return

#     # 3. Find marker indices
#     start_index = -1
#     end_index = -1

#     for i, line in enumerate(target_lines):
#         if START_MARKER in line:
#             start_index = i
#         elif END_MARKER in line:
#             end_index = i
#             break  # Assuming END_MARKER always comes after START_MARKER

#     if start_index == -1:
#         print(f"Error: '{START_MARKER}' not found in '{target_file_path}'.")
#         return
#     if end_index == -1:
#         print(f"Error: '{END_MARKER}' not found in '{target_file_path}'.")
#         return
#     if start_index >= end_index:
#         print(
#             f"Error: Markers in '{target_file_path}' are in the wrong order or on the same line."
#         )
#         return

#     # 4. Construct the new content
#     # Lines before the START_MARKER (inclusive)
#     pre_block = target_lines[: start_index + 1]
#     # The new parameter definitions (as a list of one string, which is the whole block)
#     new_block_lines = [new_params_content]
#     # Lines from the END_MARKER (inclusive)
#     post_block = target_lines[end_index:]

#     new_full_content_lines = pre_block + new_block_lines + post_block

#     insert_line = f"print('Init ML model for {challenge_name.upper()} challenge')"

#     # New logic to find the marker and insert the line
#     DEF_INIT_MARKER = "def __init__(self, metadata):"
#     found_marker_index = -1
#     marker_indentation_str = ""

#     for idx, current_line_in_file in enumerate(new_full_content_lines):
#         stripped_line = current_line_in_file.lstrip()
#         if stripped_line.startswith(DEF_INIT_MARKER):
#             found_marker_index = idx
#             # Calculate indentation from the original line
#             num_leading_spaces = len(current_line_in_file) - len(stripped_line) + 4 # +4 for the indentation after __init__
#             marker_indentation_str = " " * num_leading_spaces
#             break

#     if found_marker_index != -1:
#         # Prepare the line to be inserted with the marker's indentation
#         line_to_add = marker_indentation_str + insert_line
#         if not line_to_add.endswith("\n"):
#             line_to_add += "\n"

#         # Insert the new line *after* the marker line
#         new_full_content_lines.insert(found_marker_index + 1, line_to_add)
#         print(
#             f"Successfully inserted line after '{DEF_INIT_MARKER}' in the content."
#         )
#     else:
#         print(
#             f"Warning: Marker '{DEF_INIT_MARKER}' not found. "
#             f"Skipping insertion of: {insert_line}"
#         )

#     # 5. Write the changes to the target file
#     try:
#         with open(target_file_path, "w") as f:
#             f.writelines(new_full_content_lines)
#         print(
#             f"Successfully updated hyperparameters in '{target_file_path}' from '{new_params_file_path}'."
#         )
#     except IOError as e:
#         print(f"Error writing to target file '{target_file_path}': {e}")



# def update_ramp_kit_hyperparameters(
#     prediction_type,
#     ramp_kit_dir,
#     submission,
#     data_name,
#     model_used="lgbm",
#     # TODO: error here !!!! 
#     ramp_kits_path=".",
#     hps_path=".",
#     hps_text=HPS_TEXT
# ):
#     """
#     Update hyperparameters of a model in a RAMP kit.
    
#     Parameters
#     ----------
#     prediction_type : str
#         Type of prediction task ('regression' or 'classification')
#     ramp_kit_dir : str
#         Directory name of the RAMP kit
#     submission : str
#         Name of the submission folder
#     data_name : str
#         Name of the data/challenge
#     model_used : str, default='lgbm'
#         Model type to update (e.g., 'lgbm', 'rf', etc.)
#     ramp_kits_path : str, default=RAMP_KITS_PATH
#         Path to the RAMP kits directory
#     hps_path : str, default=HPS_PATH
#         Path to the hyperparameters directory
#     hps_text : str, default=HPS_TEXT
#         Name of the hyperparameters file
    
#     Returns
#     -------
#     None
#     """
#     prediction_type = get_prediction_type(prediction_type)
#     # get model file name to modify 
#     model_file_name = "regressor" if prediction_type == "regression" else "classifier"
    
#     target_model_hps_path = os.path.join(ramp_kits_path, ramp_kit_dir, "submissions", submission, f"{model_file_name}.py")
#     challenge_hps_py_path = os.path.join(hps_path, model_used, data_name, hps_text)
    
#     print(f"\nUpdating HPs for {target_model_hps_path} using {challenge_hps_py_path}")
#     if not os.path.exists(challenge_hps_py_path):
#         raise FileNotFoundError(f"Hyperparameter file not found: {challenge_hps_py_path}")

#     update_hyperparameters(
#         target_file_path=target_model_hps_path,
#         new_params_file_path=challenge_hps_py_path,
#         challenge_name=data_name,
#     )
#     print(f"Hyperparameters updated for {data_name} challenge.")


