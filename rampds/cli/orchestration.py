import glob
import json
import shutil
import random
import datetime
import numpy as np
import pandas as pd
import rampwf as rw
import rampds as rs
from pathlib import Path
from typing import Optional

def last_action(ramp_kit_dir, name):
    """Last action of a given action name."""
    action_f_names = glob.glob(f'{ramp_kit_dir}/actions/*')
    action_f_names.sort(reverse=True)
    for i in range(len(action_f_names)):
        ramp_action_object = rs.actions.load_ramp_action(action_f_names[i])
        if ramp_action_object.name == name:
            return ramp_action_object
    return None


@rs.actions.ramp_action
def kaggle_submit_file(
    submission_source_f_name: str,
    submission_target_f_name: str,
    ramp_kit_dir: str,
):
    """Copy a submission file into a submission folder.

    Typically into <ramp_kit_dir>/kaggle_submissions.
    Can be used outside Kaggle but the name stuck.
    The main function of this is to save the action so we can recover what was
    submitted.
    """
    shutil.copy(submission_source_f_name, submission_target_f_name)


def run_race(
    base_submissions: list[str],
    action_stats: dict,
    ramp_kit_dir: str,
    kit_suffix: str,
    metadata: dict,
    n_rounds: int,
    n_trials_per_round: int,
    patience: int,
    n_folds_hyperopt: int,
    start_round: int,
    scores: list[float],
    is_lower_the_better: bool,
    contributivity_floor: int,
    blended_submissions: set[str],
) -> set[str]:
    # can be deleted, once the algorithm settles
    improvement_speed_df = pd.DataFrame(columns=["round"] + base_submissions)
    
    for round_idx in range(start_round, n_rounds):
        if patience >= 0 and len(scores) > patience:
            if is_lower_the_better:
                if min(scores[:-patience]) <= min(scores):
                    break
            else:
                if max(scores[:-patience]) >= max(scores):
                    break
        # We'll choose the submission that improves the results the fastest
        improvement_speeds = {}
        for submission in base_submissions:
            ast = action_stats[submission]
            # sum of contributivities, applying this action
            if len(ast) >= 2:
                last_ast = ast[-1]
                total_time = np.array([a["runtime"].total_seconds() for a in ast]).sum()
                contributivity = last_ast["contributivities"][submission]
                speed = contributivity / total_time
            else:
                speed = np.finfo(float).max            
            improvement_speeds[submission] = speed
        print(f'improvement_speeds:\n{improvement_speeds}')
        row = pd.DataFrame(dict({"round_idx" : round_idx}, **improvement_speeds), index=[round_idx])
        improvement_speed_df = pd.concat([improvement_speed_df, row], ignore_index=True)
        improvement_speed_df.to_csv("improvement_speeds.csv")
        max_speed = max(improvement_speeds.values())
        best_submissions = [submission for submission, speed in improvement_speeds.items() if speed == max_speed]
        submission = random.choice(best_submissions)  # in case of tie (at zero typically), or eps greedy, random choice
        print(f'best_submissions: {best_submissions}')
        print(f'selected submission : {submission}')
    #    input("Press Enter to continue...")
        n_trials = n_trials_per_round * n_folds_hyperopt
        if "regression" in metadata["prediction_type"]:
            workflow_element_names = ["regressor"]
        elif "classification" in metadata["prediction_type"]:
            workflow_element_names = ["classifier"]
        rs.actions.hyperopt(
            ramp_kit_dir = ramp_kit_dir,
            submission = submission,
            workflow_element_names = workflow_element_names,
            n_trials = n_trials,
            fold_idxs = range(900, 900 + n_folds_hyperopt),
            resume = True,
            subtract_existing = False,
        )
        hyperopt_action = last_action(ramp_kit_dir, "hyperopt")
        if len(hyperopt_action.mean_scores) > 0:
            # Add the best submission from this round of hyperopt to the set to be blended
            for hyperopt_submission, mean_score in hyperopt_action.mean_scores.items():
                # Add best
                #if mean_score == hyperopt_action.mean_score:
                blended_submissions.add(hyperopt_submission)
        
            # The blended score improvement is wrt the previous blended score. If it doesn't exist
            # (in the first iteration, or if no submission was blended for a reason) use the mean
            # score.
            previous_blend_action = last_action(ramp_kit_dir, "blend")
            rs.actions.blend(
                ramp_kit_dir = ramp_kit_dir,
                submissions = list(blended_submissions),
                fold_idxs = range(900, 900 + n_folds_hyperopt),
            )
            blend_action = last_action(ramp_kit_dir, "blend")
            if hasattr(blend_action, "blended_score"):
                blended_score = blend_action.blended_score
                contributivities = {
                    s: contributivity_floor + np.array([c for sh, c in blend_action.contributivities.items() if sh[:len(s)] == s]).sum()
                    for s in base_submissions
                }
                submission_source_f_name = Path(ramp_kit_dir) / "submissions" / "training_output" / "submission_combined_bagged_test.csv"
                submission_target_f_name = Path(ramp_kit_dir) / "kaggle_submissions" / f"auto_{kit_suffix}_{str(round_idx).zfill(3)}.csv"
                kaggle_submit_file(
                    submission_source_f_name = submission_source_f_name,
                    submission_target_f_name = submission_target_f_name,
                    ramp_kit_dir = ramp_kit_dir,
                )
            else:
                print("something wrong: no belnded score")
                exit(0)
                blended_score = hyperopt_action.mean_score
                contributivities = {s: 1000 / len(submissions) for s in base_submissions}
            scores.append(blended_score)
            action_stats[submission].append({
                "runtime": hyperopt_action.runtime,
                "contributivities": contributivities,
            })
        
            # Delete submissions with zero contributivity from the submissions to be blended
            contributivities_df = rs.actions.load_contributivities(ramp_kit_dir)
            non_contributive_submissions = set(contributivities_df[contributivities_df["contributivity"] == 0].index)
            print(f'blended submissions: {blended_submissions}')
            print(f'non-contributive submissions: {non_contributive_submissions}')
            blended_submissions = blended_submissions - non_contributive_submissions
            print(f'new blended submissions: {blended_submissions}')
        #    input("Press Enter to continue...")
    return blended_submissions


def resume_race(
    action_stats: dict,
    ramp_kit_dir: str,
    base_submissions: list[str],
    contributivity_floor: int,
) -> tuple[int, set[str], dict, list[float]]:
    print("Loading actions...")
    action_f_names = glob.glob(f'{ramp_kit_dir}/actions/*')
    action_f_names.sort()
    ramp_program = []
    for action_f_name in action_f_names:
        f_name = Path(action_f_name).name
        ramp_program.append(rs.actions.load_ramp_action(action_f_name))
    blend_actions = [ra for ra in ramp_program if ra.name == "blend"]
    stop_time = blend_actions[-1].stop_time
    print(f"Last blending action at {stop_time}, deleting all actions after...")
    actions_f_names_to_delete = [a for a in action_f_names if pd.to_datetime(Path(a).stem) > stop_time]
    for f_name in actions_f_names_to_delete:
        Path(f_name).unlink()
    print("Re-loading actions...")
    action_f_names = glob.glob(f'{ramp_kit_dir}/actions/*')
    action_f_names.sort()
    ramp_program = []
    for action_f_name in action_f_names:
        f_name = Path(action_f_name).name
        ramp_program.append(rs.actions.load_ramp_action(action_f_name))
    blend_actions = [ra for ra in ramp_program if ra.name == "blend"]
    hyperopt_actions = [ra for ra in ramp_program if ra.name == "hyperopt"]
    start_round = len(hyperopt_actions)
    scores = []
    print(f"Recovering {len(hyperopt_actions)} hyperopt and blend actions...")
    blend_action_idx = 0
    for hyperopt_action in hyperopt_actions:
        submission = hyperopt_action.kwargs["submission"]
        if len(hyperopt_action.mean_scores) > 0:
            blend_action = blend_actions[blend_action_idx]
            blend_action_idx += 1  # if hyperopt did not return with any submissions, there was no blend
            blended_score = blend_action.blended_score
            scores.append(blended_score)
            contributivities = {
                s: contributivity_floor + np.array([c for sh, c in blend_action.contributivities.items() if sh[:len(s)] == s]).sum()
                for s in base_submissions
            }
            action_stats[submission].append({
                "runtime": hyperopt_action.runtime,
                "contributivities": contributivities,
            })
            blended_submissions = set([sh for sh, c in blend_action.contributivities.items() if c > 0])
    print(f'Blended submissions: {blended_submissions}')
    return start_round, blended_submissions, action_stats, scores


def final_blend_growing_folds(
    base_submissions: list[str],
    ramp_kit_dir: str,
    kit_suffix: str,
    n_folds: int,
    n_folds_hyperopt: int,
    top_n_for_mean: int,
    n_sigma: float,
):
    """Select best of each base submission (hyperopt) within n sigma and blend.

    We'll probably drop this since blend_and_bag and bag_and_blend work better,
    but keep it for early versions to experiment.
    """
    for stop_fold_idx in range(900 + n_folds_hyperopt + 1, 900 + n_folds + 1):
        selected_submissions = []
        for submission in base_submissions:
            rs.actions.select_top_hyperopt_and_train(
                ramp_kit_dir = ramp_kit_dir,
                submission = submission,
                fold_idxs = range(900, stop_fold_idx),
                trained_fold_idxs = range(900, stop_fold_idx - 1),
                top_n = top_n_for_mean,
                n_sigma = n_sigma,
            )
            top_hyperopt_dict = rs.actions.select_top_hyperopt(
                ramp_kit_dir = ramp_kit_dir,
                submission = submission,
                fold_idxs = range(900, stop_fold_idx),
                n_sigma = n_sigma,
            )
            if "selected_submissions" in top_hyperopt_dict:
                selected_submissions += top_hyperopt_dict["selected_submissions"]
        rs.actions.blend(
            ramp_kit_dir = ramp_kit_dir,
            submissions = selected_submissions,
            fold_idxs = range(900, stop_fold_idx),
        )
        submission_source_f_name = Path(ramp_kit_dir) / "submissions" / "training_output" / "submission_combined_bagged_test.csv"
        submission_target_f_name =  Path(ramp_kit_dir) / "kaggle_submissions" / f"auto_{kit_suffix}_growing_folds_{str(stop_fold_idx).zfill(3)}.csv"
        kaggle_submit_file(
            submission_source_f_name = submission_source_f_name,
            submission_target_f_name = submission_target_f_name,
            ramp_kit_dir = ramp_kit_dir,
        )


def update_hyperopt_summary(
    base_submissions: list[str],
    ramp_kit_dir: str,
):
    """Update the hyperopt summary.

    This is needed when submissions are trained outside hyperopt.
    """
    for submission in base_submissions:
        rs.actions.update_hyperopt_score_summary(
            ramp_kit_dir = ramp_kit_dir,
            submission = submission,
        )


def train_on_all_folds(
    submissions: list[str],
    ramp_kit_dir: str,
    n_folds: int,
):
    """We retrain the final blend on all folds.
    """
    for submission in submissions:
        rs.actions.train(
            ramp_kit_dir = ramp_kit_dir,
            submission = submission,
            fold_idxs = range(900, 900 + n_folds),
        )   


def final_blend_then_bag(
    submissions: list[str],
    ramp_kit_dir: str,
    kit_suffix: str,
    n_folds: int,
    n_rounds: Optional[int] = -1, 
):
    """Blend then bag and submit after each fold.

    To potentially recover the learning curve. Typically we only submit the last one,
    but we save all in <ramp_kit_dir>/kaggle_submissions.
    """
    for stop_fold_idx in range(901, 900 + n_folds + 1):        
        rs.actions.blend(
            ramp_kit_dir = ramp_kit_dir,
            submissions = submissions,
            fold_idxs = range(900, stop_fold_idx),
        )
        submission_source_f_name = Path(ramp_kit_dir) / "submissions" / "training_output" / "submission_combined_bagged_test.csv"
        if n_rounds > 0:
            submission_target_f_name = Path(ramp_kit_dir) / "kaggle_submissions" / f"auto_{kit_suffix}_last_blend_{str(stop_fold_idx).zfill(3)}_r{n_rounds}.csv"
        else:
            submission_target_f_name = Path(ramp_kit_dir) / "kaggle_submissions" / f"auto_{kit_suffix}_last_blend_{str(stop_fold_idx).zfill(3)}.csv"
        kaggle_submit_file(
            submission_source_f_name = submission_source_f_name,
            submission_target_f_name = submission_target_f_name,
            ramp_kit_dir = ramp_kit_dir,
        )


def final_bag_then_blend(
    submissions: list[str],
    ramp_kit_dir: str,
    kit_suffix: str,
    n_folds: int,
    n_rounds: Optional[int] = -1, 
):
    """Bag then blend and submit after each fold.

    To potentially recover the learning curve. Typically we only submit the last one,
    but we save all.
    """
    for stop_fold_idx in range(901, 900 + n_folds + 1):        
        rs.actions.bag_then_blend(
            ramp_kit_dir = ramp_kit_dir,
            submissions = submissions,
            fold_idxs = range(900, stop_fold_idx),
        )
        submission_source_f_name = Path(ramp_kit_dir) / "submissions" / "training_output" / "submission_bagged_then_blended_test.csv"
        if n_rounds > 0:
            submission_target_f_name = Path(ramp_kit_dir) / "kaggle_submissions" / f"auto_{kit_suffix}_bagged_then_blended_{str(stop_fold_idx).zfill(3)}_r{n_rounds}.csv"
        else:
            submission_target_f_name = Path(ramp_kit_dir) / "kaggle_submissions" / f"auto_{kit_suffix}_bagged_then_blended_{str(stop_fold_idx).zfill(3)}.csv"
        kaggle_submit_file(
            submission_source_f_name = submission_source_f_name,
            submission_target_f_name = submission_target_f_name,
            ramp_kit_dir = ramp_kit_dir,
        )


def submit_best_submissions(
    base_submissions: list[str],
    ramp_kit_dir: str,
    kit_suffix: str,
    n_folds: int,
):
    """Submit the best of each submission (classical hyperopt) in
    <ramp_kit_dir>/kaggle_submissions.
    """
    for submission in base_submissions:
        best_submissions = rs.actions.select_top_hyperopt(
            ramp_kit_dir = ramp_kit_dir,
            submission = submission,
            fold_idxs = range(900, 900 + n_folds),
            top_n = 1,
        )["selected_submissions"]
        if len(best_submissions) == 0:
            return
        best_submission = best_submissions[0]
        submission_source_f_name = Path(ramp_kit_dir) / "submissions" / best_submission / "training_output" / "submission_bagged_test.csv"
        submission_target_f_name = Path(ramp_kit_dir) / "kaggle_submissions" / f"auto_{kit_suffix}_best_{submission}.csv"
        kaggle_submit_file(
            submission_source_f_name = submission_source_f_name,
            submission_target_f_name = submission_target_f_name,
            ramp_kit_dir = ramp_kit_dir,
        )


def hyperopt_race(
    ramp_kit: str,
    kit_root: str,
    version: str,
    number: str | int,
    resume: bool,
    n_rounds: Optional[int] = 100,
    n_trials_per_round: Optional[int] = 5,
    patience: Optional[int] = -1,
    n_folds_hyperopt: Optional[int] = 3,
    n_folds: Optional[int] = 31,
    base_submissions: Optional[list[str]] = ["lgbm", "xgboost", "catboost"],
    top_n_for_mean: Optional[int] = 10,
    n_sigma: Optional[float] = 1.0,    
    contributivity_floor: Optional[int] = 100,  # on 1000, added to contributivity to give a chance to every submission
    no_growing_folds: Optional[bool] = False,
):
    kit_suffix = f"v{version}_n{number}"
    ramp_kit_dir = Path(kit_root) / f"{ramp_kit}_{kit_suffix}"
    problem = rw.utils.assert_read_problem(ramp_kit_dir)
    score_names = [st.name for st in problem.score_types] 
    valid_score_name = f'valid_{score_names[0]}'
    is_lower_the_better = problem.score_types[0].is_lower_the_better
    with open(ramp_kit_dir / "data" / "metadata.json", "r") as f:
        metadata = json.load(f)

    # Dictionary of submissions: list of dictionary of run times and scores
    action_stats = {submission: [] for submission in base_submissions}
    if resume:
        start_round, blended_submissions, action_stats, scores = resume_race(
            action_stats = action_stats,
            ramp_kit_dir = ramp_kit_dir,
            base_submissions = base_submissions,
            contributivity_floor = contributivity_floor,
        )
    else:
        start_round = 0
        blended_submissions = set()
        scores = []
        # submit base submissions
        for submission in base_submissions:
            if "regression" in metadata["prediction_type"]:
                rs.scripts.tabular.tabular_regression_columnwise_first_submit(         
                    ramp_kit_dir = ramp_kit_dir,
                    submission = submission,
                    regressor = submission,
                )
            elif "classification" in metadata["prediction_type"]:
                rs.scripts.tabular.tabular_classification_columnwise_first_submit(         
                    ramp_kit_dir = ramp_kit_dir,
                    submission = submission,
                    classifier = submission,
                )
        kaggle_submissions_path = ramp_kit_dir / "kaggle_submissions"
        kaggle_submissions_path.mkdir(parents=False, exist_ok=True)
          
    blended_submissions = run_race(
        base_submissions = base_submissions,
        action_stats = action_stats,
        ramp_kit_dir = ramp_kit_dir,
        kit_suffix = kit_suffix,
        metadata = metadata,
        n_rounds = n_rounds,
        n_trials_per_round = n_trials_per_round,
        patience = patience,
        n_folds_hyperopt = n_folds_hyperopt,
        start_round = start_round,
        scores = scores,
        is_lower_the_better = is_lower_the_better,
        contributivity_floor = contributivity_floor,
        blended_submissions = blended_submissions,
    )

    # Run the growing folds algorithm: select best of each base submission within
    # n sigma and train on one more fold, then repeat. Then blend the best ones.
    # Conservative blend (discard diverse but bad submissions), we'll probably drop
    # this since blend_and_bag and bag_and_blend work better, but keep it for early
    # versions to experiment.
    if not no_growing_folds:
        final_blend_growing_folds(
            base_submissions = base_submissions,
            ramp_kit_dir = ramp_kit_dir,
            kit_suffix = kit_suffix,
            n_folds = n_folds,
            n_folds_hyperopt = n_folds_hyperopt,
            top_n_for_mean = top_n_for_mean,
            n_sigma = n_sigma,
        )
    # Train the final blend of the hyperopt race on all the folds
    train_on_all_folds(
        submissions = list(blended_submissions),
        ramp_kit_dir = ramp_kit_dir,
        n_folds = n_folds,
    )
    # Whenever hyperopt submissions are trained outside of hyperopt, we need to update the summary.
    update_hyperopt_summary(
        base_submissions = base_submissions,
        ramp_kit_dir = ramp_kit_dir,
    )
    # Blend then bag the final blend of the hyperopt race on all the folds
    final_blend_then_bag(
        submissions = list(blended_submissions),
        ramp_kit_dir = ramp_kit_dir,
        kit_suffix = kit_suffix,
        n_folds = n_folds,
    )
    # Bag then blend the final blend of the hyperopt race on all the folds
    final_bag_then_blend(
        submissions = list(blended_submissions),
        ramp_kit_dir = ramp_kit_dir,
        kit_suffix = kit_suffix,
        n_folds = n_folds,
    )
    # Submit the best of each base submission (classical hyperopt)
    submit_best_submissions(
        base_submissions = base_submissions,
        ramp_kit_dir = ramp_kit_dir,
        kit_suffix = kit_suffix,
        n_folds = n_folds,
    )
