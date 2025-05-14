import glob
import json
import random
import shutil
from pathlib import Path
from typing import Optional, List
import numpy as np
import pandas as pd
import rampds as rs
import rampwf as rw
import logging
logger = logging.getLogger("orchestration")

def last_action(
    ramp_kit_dir: str,
    name: str,
    fold_idxs: range = None,
) -> rs.actions.RampAction | None:
    """Last action of a given action name."""
    action_f_names = glob.glob(f"{ramp_kit_dir}/actions/*")
    action_f_names.sort(reverse=True)
    for i in range(len(action_f_names)):
        ramp_action_object = rs.actions.load_ramp_action(Path(action_f_names[i]))
        if ramp_action_object.name == name:
            if fold_idxs is None or ramp_action_object.kwargs["fold_idxs"] == fold_idxs:
                return ramp_action_object
    return None


@rs.actions.ramp_action
def submit_final_test_predictions(
    submission_source_f_name: str,
    submission_target_f_name: str,
    ramp_kit_dir: str,
):
    """Copy a submission file into a submission folder.

    Typically into <ramp_kit_dir>/final_test_predictions.
    The main function of this is to save the action so we can recover what was
    submitted.
    """
    shutil.copy(submission_source_f_name, submission_target_f_name)

class Orchestrator():
    def __init__(self, ramp_kit_dir, base_predictors, preprocessors_to_hyperopt, contributivity_floor, patience,
                 is_lower_the_better, metadata, n_rounds, max_time, race_blend, n_folds_hyperopt, n_folds_final_blend,
                 n_trials_per_round, first_fold_idx, n_cpu_per_run):
        self.ramp_kit_dir = ramp_kit_dir
        self.base_predictors = base_predictors
        self.preprocessors_to_hyperopt = preprocessors_to_hyperopt
        self.contributivity_floor = contributivity_floor
        self.patience = patience
        self.is_lower_the_better = is_lower_the_better
        self.metadata = metadata
        self.n_rounds = n_rounds
        self.max_time = max_time
        self.race_blend = race_blend
        self.n_folds_hyperopt = n_folds_hyperopt
        self.n_folds_final_blend = n_folds_final_blend
        self.n_trials_per_round = n_trials_per_round
        self.first_fold_idx = first_fold_idx
        self.n_cpu_per_run = n_cpu_per_run
        
        self.base_predictor_stats = {submission: [] for submission in self.base_predictors}
        self.scores = []
        self.blended_submissions = set()
        self.round_idx = 0
        self.elapsed_time = 0.0
        self.estimated_final_blending_time = 0.0
        self.final_round_datetime = None
        self.fold_idxs = range(first_fold_idx, first_fold_idx + n_folds_hyperopt)
        
    def update_with_actions(self, hyperopt_action, blend_action):      
        submission = hyperopt_action.kwargs["submission"]
        base_predictor = submission
        if "hyperopt" in submission:
            base_predictor = submission[:-20]  # len(hyperopt_<hash_of_len_10>) = 19

        if hasattr(blend_action, "blended_score"):
            blended_score = blend_action.blended_score
            # Add up contributivites belonging to base predictors in the current blend
            base_contributivities = {
                s: self.contributivity_floor
                + np.array([c for sh, c in blend_action.contributivities.items() if sh[: len(s)] == s]).sum()
                for s in self.base_predictors
            }
        else:
            print("something wrong: no blended score")
            raise RuntimeError("something wrong: no blended score")
        self.round_idx += 1
        self.elapsed_time += hyperopt_action.runtime.total_seconds() / 3600
        self.elapsed_time += blend_action.runtime.total_seconds() / 3600
        self.scores.append(blend_action.blended_score)
        self.base_predictor_stats[base_predictor].append(
            {
                "runtime": hyperopt_action.runtime,
                "contributivities": base_contributivities,
            }
        )
        self.blended_submissions = set([sh for sh, c in blend_action.contributivities.items() if c > 0])
        print(f"Blended submissions: {self.blended_submissions}")
        self.estimated_final_blending_time = 2 * blend_action.runtime.total_seconds() * self.n_folds_final_blend / self.n_folds_hyperopt / 3600
        self.final_round_datetime = blend_action.start_time

    def stop(self):
        if self.round_idx >= self.n_rounds:
            print(f"Stopping for reaching max rounds = {self.n_rounds}")
            return True
        if self.patience >= 0 and len(self.scores) > self.patience:
            if self.is_lower_the_better:
                print(f"Best occured {len(self.scores) - self.scores.index(min(self.scores))} rounds ago.")
                if min(self.scores[:-self.patience]) <= min(self.scores):
                    print(f"Stopping for reaching patience = {self.patience} at round {self.round_idx}")
                    return True
            else:
                print(f"Best occured {len(self.scores) - self.scores.index(max(self.scores))} rounds ago.")
                if max(self.scores[:-self.patience]) >= max(self.scores):
                    print(f"Stopping for reaching patience = {self.patience} at round {self.round_idx}")
                    return True    
        estimated_runtime_for_final_blend = 0
        for submission in self.blended_submissions:
            scores_df = pd.read_csv(f"{self.ramp_kit_dir}/submissions/{submission}/training_output/fold_{self.first_fold_idx}/scores.csv")
            estimated_runtime_for_final_blend += scores_df["time"].sum()
        estimated_runtime_for_final_blend *= (self.n_folds_final_blend - self.n_folds_hyperopt) / 3600
        if self.elapsed_time + estimated_runtime_for_final_blend + self.estimated_final_blending_time > self.max_time:
            print("Stopping for time limit:")
            print(f"Elapsed time: {self.elapsed_time:.2f} hours")
            print(f"\nEstimated runtime (train + valid + test) for final blend: {estimated_runtime_for_final_blend:.2f} hours")
            print(f"\nEstimated final blending time: {self.estimated_final_blending_time:.2f} hours")
            return True
        return False

    def choose_base_predictor(self):
        # We'll choose the submission that improves the results the fastest        
        improvement_speeds = {}
        for base_predictor in self.base_predictors:
            bpst = self.base_predictor_stats[base_predictor]
            # sum of contributivities, applying this action
            if len(bpst) >= 2:
                last_bpst = bpst[-1]
                total_time = np.array([a["runtime"].total_seconds() for a in bpst]).sum()
                contributivity = last_bpst["contributivities"][base_predictor]
                speed = contributivity / total_time
            else:
                speed = np.finfo(float).max
            improvement_speeds[base_predictor] = speed
        print(f"improvement_speeds:\n{improvement_speeds}")

        max_speed = max(improvement_speeds.values())
        best_base_predictors = [bp for bp, speed in improvement_speeds.items() if speed == max_speed]
        base_predictor = random.choice(best_base_predictors)  # in case of tie (at zero typically), or eps greedy, random choice
        print(f"best base predictors: {best_base_predictors}")
        print(f"selected base predictor : {base_predictor}")
        #    input("Press Enter to continue...")
        return base_predictor

    def choose_workflow_elements(self, base_predictor):
        if "regression" in self.metadata["prediction_type"]:
            we_names = ["regressor"]
        elif "classification" in self.metadata["prediction_type"]:
            we_names = ["classifier"]
        if self.preprocessors_to_hyperopt is not None:
            we_names += self.preprocessors_to_hyperopt
        bpst = self.base_predictor_stats[base_predictor]
        blended_submissions_of_predictor = [
            submission for submission in self.blended_submissions if submission[:-20] == base_predictor
        ]
        if len(bpst) >= 2 and len(blended_submissions_of_predictor) > 0:
            # if we just start the race, or there is no base submission in the blend
            # do not optimize the preprocessor, rather use the default
            # this makes sure that eg we can control the default, eg which features
            # are dropped.
            # This has to be rethought, because if one predictor is already very good with a
            # non-default preprocessor, others cannot enter the blend. We should hybridize
            # base models and preprocessors.
            wes_to_hyperopt = random.sample(we_names, 1)
        else:
            wes_to_hyperopt = [we_names[0]]
        print(f"Workflow elements to choose from: {we_names}")
        print(f"Choosen workflow element: {wes_to_hyperopt[0]}")
        return wes_to_hyperopt

    def choose_submission(self, base_predictor):
        submission_to_hyperopt = base_predictor  # default: hyperopt one of the base submissions
        blended_submissions_of_predictor = [
            submission for submission in self.blended_submissions if submission[:-20] == base_predictor
        ]
        print(f"Submissions to choose from: {blended_submissions_of_predictor}")
        if len(blended_submissions_of_predictor) > 0:
            submission_to_hyperopt = random.choice(blended_submissions_of_predictor)
        print(f"Choosen submission: {submission_to_hyperopt}")
        return submission_to_hyperopt

def run_race(
    orchestrator: Orchestrator,
    ramp_kit_dir: str,
) -> Orchestrator:
    logger.info(f"Race starting at round {orchestrator.round_idx}")
    while not orchestrator.stop():
        base_predictor = orchestrator.choose_base_predictor()
        wes_to_hyperopt = orchestrator.choose_workflow_elements(base_predictor)
        submission_to_hyperopt = orchestrator.choose_submission(base_predictor)  
        rs.actions.hyperopt(
            ramp_kit_dir=ramp_kit_dir,
            submission=submission_to_hyperopt,
            workflow_element_names=wes_to_hyperopt,
            n_trials=orchestrator.n_trials_per_round * orchestrator.n_folds_hyperopt,
            fold_idxs=orchestrator.fold_idxs,
            resume=True,
            subtract_existing=False,
            n_cpu_per_run=orchestrator.n_cpu_per_run
        )
        # we read the action from file because it also contains timing information
        hyperopt_action = last_action(ramp_kit_dir, "hyperopt")
        if not hyperopt_action is None and len(hyperopt_action.mean_scores) > 0:
            # Add all submissions from this round of hyperopt to the set to be blended
            submissions_to_blend = orchestrator.blended_submissions.copy()
            for hyperopt_submission, mean_score in hyperopt_action.mean_scores.items():
                submissions_to_blend.add(hyperopt_submission)
            if orchestrator.race_blend == "blend":
                rs.actions.blend(
                    ramp_kit_dir=ramp_kit_dir,
                    submissions=list(submissions_to_blend),
                    fold_idxs=orchestrator.fold_idxs,
                )
            elif orchestrator.race_blend == "bag_then_blend":
                rs.actions.bag_then_blend(
                    ramp_kit_dir=ramp_kit_dir,
                    submissions=list(submissions_to_blend),
                    fold_idxs=orchestrator.fold_idxs,
                )
            else:
                raise ValueError(f"{orchestrator.race_blend}: unknown blending strategy")
            blend_action = last_action(ramp_kit_dir, orchestrator.race_blend, fold_idxs=orchestrator.fold_idxs)
            orchestrator.update_with_actions(hyperopt_action, blend_action)
        #    input("Press Enter to continue...")
    return orchestrator


def resume_race(
    orchestrator: Orchestrator,
    ramp_kit_dir: str,
) -> Orchestrator:
    print("Loading actions...")
    action_f_names = glob.glob(f"{ramp_kit_dir}/actions/*")
    action_f_names.sort()
    ramp_program = []
    for action_f_name in action_f_names:
        f_name = Path(action_f_name).name
        ramp_program.append(rs.actions.load_ramp_action(Path(action_f_name)))
    blend_actions = [ra for ra in ramp_program if ra.name == "blend" and ra.kwargs["fold_idxs"] == orchestrator.fold_idxs]
    
    if len(blend_actions) == 0:
        # Crash occured early, before the first round
        start_round = 0
        return orchestrator

    stop_time = blend_actions[-1].stop_time
    print(f"Last blending action at {stop_time}, deleting all actions after...")
    actions_f_names_to_delete = [a for a in action_f_names if pd.to_datetime(Path(a).stem) > stop_time]
    for f_name in actions_f_names_to_delete:
        Path(f_name).unlink()
    print("Re-loading actions...")
    action_f_names = glob.glob(f"{ramp_kit_dir}/actions/*")
    action_f_names.sort()
    ramp_program = []
    for action_f_name in action_f_names:
        f_name = Path(action_f_name).name
        ramp_program.append(rs.actions.load_ramp_action(Path(action_f_name)))
    # we only need race blend actions
    blend_actions = [ra for ra in ramp_program if ra.name == "blend" and ra.kwargs["fold_idxs"] == orchestrator.fold_idxs]
    hyperopt_actions = [ra for ra in ramp_program if ra.name == "hyperopt"]
    print(f"Recovering {len(hyperopt_actions)} hyperopt and blend actions...")
    blend_action_idx = 0
    for hyperopt_action in hyperopt_actions:
        if len(hyperopt_action.mean_scores) > 0:
            try:
                blend_action = blend_actions[blend_action_idx]
            except IndexError:
                print(f"blend_actions[{blend_action_idx}] does not exist, possibly corrupted actions")
                blend_action_idx += 1
                continue
            blend_action_idx += 1  # if hyperopt did not return with any submissions, there was no blend
            orchestrator.update_with_actions(hyperopt_action, blend_action)
    return orchestrator


def update_hyperopt_summary(
    base_predictors: list[str],
    ramp_kit_dir: str,
):
    """Update the hyperopt summary.

    This is needed when submissions are trained outside hyperopt.
    """
    for submission in base_predictors:
        rs.actions.update_hyperopt_score_summary(
            ramp_kit_dir=ramp_kit_dir,
            submission=submission,
        )


def train_on_all_folds(
    submissions: list[str],
    ramp_kit_dir: str,
    n_folds_final_blend: int,
    first_fold_idx: int,
):
    """We retrain the final blend on all folds."""
    for submission in submissions:
        rs.actions.train(
            ramp_kit_dir=ramp_kit_dir,
            submission=submission,
            fold_idxs=range(first_fold_idx, first_fold_idx + n_folds_final_blend),
        )


@rs.actions.ramp_action
def final_blend_then_bag(
    submissions: list[str],
    ramp_kit_dir: str,
    kit_suffix: str,
    n_folds_final_blend: int,
    first_fold_idx: int,
    n_rounds: int = -1,
    round_datetime = None,  # for recording in the action, not used in the function
):
    """Blend then bag and submit after each fold.

    To potentially recover the learning curve. Typically we only submit the last one,
    but we save all in <ramp_kit_dir>/final_test_predictions.
    """
#    for stop_fold_idx in range(first_fold_idx + 1, first_fold_idx + n_folds_final_blend + 1):
    for stop_fold_idx in range(first_fold_idx + n_folds_final_blend, first_fold_idx + n_folds_final_blend + 1):
        rs.actions.blend(
            ramp_kit_dir=ramp_kit_dir,
            submissions=submissions,
            fold_idxs=range(first_fold_idx, stop_fold_idx),
        )
        submission_source_f_name = (
            Path(ramp_kit_dir) / "submissions" / "training_output" / "submission_combined_bagged_test.csv"
        )
        if n_rounds > 0:
            submission_target_f_name = (
                Path(ramp_kit_dir)
                / "final_test_predictions"
                / f"auto_{kit_suffix}_last_blend_{str(stop_fold_idx).zfill(3)}_r{n_rounds}.csv"
            )
        else:
            submission_target_f_name = (
                Path(ramp_kit_dir)
                / "final_test_predictions"
                / f"auto_{kit_suffix}_last_blend_{str(stop_fold_idx).zfill(3)}.csv"
            )
        submit_final_test_predictions(
            submission_source_f_name=str(submission_source_f_name),
            submission_target_f_name=str(submission_target_f_name),
            ramp_kit_dir=ramp_kit_dir,
        )
        submission_source_f_name = (
            Path(ramp_kit_dir) / "submissions" / "training_output" / "submission_combined_bagged_valid.csv"
        )
        if n_rounds > 0:
            submission_target_f_name = (
                Path(ramp_kit_dir)
                / "final_test_predictions"
                / f"auto_{kit_suffix}_last_blend_{str(stop_fold_idx).zfill(3)}_r{n_rounds}_valid.csv"
            )
        else:
            submission_target_f_name = (
                Path(ramp_kit_dir)
                / "final_test_predictions"
                / f"auto_{kit_suffix}_last_blend_{str(stop_fold_idx).zfill(3)}_valid.csv"
            )
        submit_final_test_predictions(
            submission_source_f_name=str(submission_source_f_name),
            submission_target_f_name=str(submission_target_f_name),
            ramp_kit_dir=ramp_kit_dir,
        )


@rs.actions.ramp_action
def final_bag_then_blend(
    submissions: list[str],
    ramp_kit_dir: str,
    kit_suffix: str,
    n_folds_final_blend: int,
    first_fold_idx: int,
    n_rounds: int = -1,
    round_datetime = None,  # for recording in the action, not used in the function
):
    """Bag then blend and submit after each fold.

    To potentially recover the learning curve. Typically we only submit the last one,
    but we save all.
    """
#    for stop_fold_idx in range(first_fold_idx + 1, first_fold_idx + n_folds_final_blend + 1):
    for stop_fold_idx in range(first_fold_idx + n_folds_final_blend, first_fold_idx + n_folds_final_blend + 1):
        rs.actions.bag_then_blend(
            ramp_kit_dir=ramp_kit_dir,
            submissions=submissions,
            fold_idxs=range(first_fold_idx, stop_fold_idx),
        )
        submission_source_f_name = (
            Path(ramp_kit_dir) / "submissions" / "training_output" / "submission_bagged_then_blended_test.csv"
        )
        if n_rounds > 0:
            submission_target_f_name = (
                Path(ramp_kit_dir)
                / "final_test_predictions"
                / f"auto_{kit_suffix}_bagged_then_blended_{str(stop_fold_idx).zfill(3)}_r{n_rounds}.csv"
            )
        else:
            submission_target_f_name = (
                Path(ramp_kit_dir)
                / "final_test_predictions"
                / f"auto_{kit_suffix}_bagged_then_blended_{str(stop_fold_idx).zfill(3)}.csv"
            )
        submit_final_test_predictions(
            submission_source_f_name=str(submission_source_f_name),
            submission_target_f_name=str(submission_target_f_name),
            ramp_kit_dir=ramp_kit_dir,
        )
        submission_source_f_name = (
            Path(ramp_kit_dir) / "submissions" / "training_output" / "submission_bagged_then_blended_valid.csv"
        )
        if n_rounds > 0:
            submission_target_f_name = (
                Path(ramp_kit_dir)
                / "final_test_predictions"
                / f"auto_{kit_suffix}_bagged_then_blended_{str(stop_fold_idx).zfill(3)}_r{n_rounds}_valid.csv"
            )
        else:
            submission_target_f_name = (
                Path(ramp_kit_dir)
                / "final_test_predictions"
                / f"auto_{kit_suffix}_bagged_then_blended_{str(stop_fold_idx).zfill(3)}_valid.csv"
            )
        submit_final_test_predictions(
            submission_source_f_name=str(submission_source_f_name),
            submission_target_f_name=str(submission_target_f_name),
            ramp_kit_dir=ramp_kit_dir,
        )


def submit_best_submissions(
    base_predictors: list[str],
    ramp_kit_dir: str,
    kit_suffix: str,
    n_folds_final_blend: int,
    first_fold_idx: int,
):
    """Submit the best of each submission (classical hyperopt) in
    <ramp_kit_dir>/final_test_predictions.
    """
    for submission in base_predictors:
        best_submissions = rs.actions.select_top_hyperopt(
            ramp_kit_dir=ramp_kit_dir,
            submission=submission,
            fold_idxs=range(first_fold_idx, first_fold_idx + n_folds_final_blend),
            top_n=1,
        )["selected_submissions"]
        if len(best_submissions) == 0:
            print(f"No best {submission}")
            continue
        best_submission = best_submissions[0]
        submission_source_f_name = (
            Path(ramp_kit_dir) / "submissions" / best_submission / "training_output" / "submission_bagged_test.csv"
        )
        if not submission_source_f_name.exists():
            print(f"Bagging {submission}")
            # It wasn't in the final blends so we need to bag it
            rs.actions.train(
                ramp_kit_dir=ramp_kit_dir,
                submission=best_submission,
                fold_idxs=range(first_fold_idx, first_fold_idx + n_folds_final_blend),
                bag=True,
            )
        submission_target_f_name = (
            Path(ramp_kit_dir) / "final_test_predictions" / f"auto_{kit_suffix}_best_{submission}.csv"
        )
        submit_final_test_predictions(
            submission_source_f_name=str(submission_source_f_name),
            submission_target_f_name=str(submission_target_f_name),
            ramp_kit_dir=ramp_kit_dir,
        )
        submission_source_f_name = (
            Path(ramp_kit_dir) / "submissions" / best_submission / "training_output" / "submission_bagged_valid.csv"
        )
        submission_target_f_name = (
            Path(ramp_kit_dir) / "final_test_predictions" / f"auto_{kit_suffix}_best_{submission}_valid.csv"
        )
        submit_final_test_predictions(
            submission_source_f_name=str(submission_source_f_name),
            submission_target_f_name=str(submission_target_f_name),
            ramp_kit_dir=ramp_kit_dir,
        )


def submit_base_submissions(
    ramp_kit_dir: str,
    metadata: dict,
    base_predictors: list[str],
    data_preprocessors: list[str],
) -> list[str]:
    for submission in base_predictors:
        if "regression" in metadata["prediction_type"]:
            submitted_elements = rs.scripts.tabular.tabular_regression_ordered_submit(
                ramp_kit_dir=ramp_kit_dir,
                submission=submission,
                regressor=submission,
                data_preprocessors=data_preprocessors,
            )

        elif "classification" in metadata["prediction_type"]:
            submitted_elements = rs.scripts.tabular.tabular_classification_ordered_submit(
                ramp_kit_dir=ramp_kit_dir,
                submission=submission,
                classifier=submission,
                data_preprocessors=data_preprocessors,
            )
    final_test_predictions_path = ramp_kit_dir / "final_test_predictions"
    final_test_predictions_path.mkdir(parents=False, exist_ok=True)
    print(submitted_elements)
    return submitted_elements


def hyperopt_race(
    ramp_kit: str,
    kit_root: str,
    version: str,
    number: str | int,
    resume: bool,
    n_rounds: int = 100,
    n_trials_per_round: int = 5,
    patience: int = -1,
    n_folds_hyperopt: int = 3,
    n_folds_final_blend: int = 30,
    first_fold_idx: int = 0,
    base_predictors: list[str] = ["lgbm", "xgboost", "catboost"],
    data_preprocessors: list[str] = [
        "drop_id",
        "drop_columns",
        "col_in_train_only",
        "base_columnwise",
        "rm_constant_col",
    ],
    preprocessors_to_hyperopt: Optional[list[str]] = None,
    race_blend: str = "blend",
    max_time: float = 1000000,
    contributivity_floor: int = 100,  # on 1000, added to contributivity to give a chance to every submission
    n_cpu_per_run: int = None,
):
    kit_suffix = f"v{version}_n{number}"
    ramp_kit_dir = Path(kit_root) / f"{ramp_kit}_{kit_suffix}"
    problem = rw.utils.assert_read_problem(str(ramp_kit_dir))
    is_lower_the_better = problem.score_types[0].is_lower_the_better
    with open(ramp_kit_dir / "data" / "metadata.json", "r") as f:
        metadata = json.load(f)

    if n_cpu_per_run is not None:
        n_cpu_per_run = int(n_cpu_per_run)

    # Dictionary of submissions: list of dictionary of run times and scores
    orchestrator = Orchestrator(
        ramp_kit_dir=ramp_kit_dir,
        base_predictors=base_predictors,
        preprocessors_to_hyperopt=preprocessors_to_hyperopt,
        contributivity_floor=contributivity_floor,
        patience=patience,
        is_lower_the_better=is_lower_the_better,
        metadata=metadata,
        n_rounds=n_rounds,
        max_time=max_time,
        race_blend=race_blend,
        n_folds_hyperopt=n_folds_hyperopt, 
        n_folds_final_blend=n_folds_final_blend,
        n_trials_per_round=n_trials_per_round,
        first_fold_idx=first_fold_idx,
        n_cpu_per_run=n_cpu_per_run,
    )
    if resume:
        orchestrator = resume_race(
            orchestrator=orchestrator,
            ramp_kit_dir=str(ramp_kit_dir),
        )
        try:
            config = rs.utils.load_config(load_path=ramp_kit_dir)
            dp_hyperopt_full_name = config["preprocessors_to_hyperopt"]
            print(dp_hyperopt_full_name)
        except FileNotFoundError:
            print("No config file, resuming with command-line parameters.")
            print("Happens only on early versions that crashed without saving the config.")
            dp_hyperopt_full_name = preprocessors_to_hyperopt
    else:
        # submit base submissions
        submitted_elements = submit_base_submissions(
            ramp_kit_dir=ramp_kit_dir,
            metadata=metadata,
            base_predictors=base_predictors,
            data_preprocessors=data_preprocessors,
        )
        dp_hyperopt_full_name = []
        if preprocessors_to_hyperopt is not None:
            for dp in preprocessors_to_hyperopt:
                dp_hyperopt_full_name += rs.utils.get_full_preprocessor_name(
                    data_preprocessor=dp, submitted_preprocessors=submitted_elements["submitted_data_preprocessors"]
                )

        rs.utils.save_config(
            ramp_kit=ramp_kit,
            kit_root=kit_root,
            version=version,
            number=number,
            resume=resume,
            n_rounds=n_rounds,
            n_trials_per_round=n_trials_per_round,
            patience=patience,
            n_folds_hyperopt=n_folds_hyperopt,
            n_folds_final_blend=n_folds_final_blend,
            first_fold_idx=first_fold_idx,
            race_blend=race_blend,
            max_time=max_time,
            base_predictors=base_predictors,
            data_preprocessors=data_preprocessors,
            preprocessors_to_hyperopt=dp_hyperopt_full_name,
            contributivity_floor=contributivity_floor,
            save_path=ramp_kit_dir,
        )

    orchestrator = run_race(
        orchestrator=orchestrator,
        ramp_kit_dir=str(ramp_kit_dir),
    )

    # Train the final blend of the hyperopt race on all the folds
    train_on_all_folds(
        submissions=list(orchestrator.blended_submissions),
        ramp_kit_dir=str(ramp_kit_dir),
        n_folds_final_blend=n_folds_final_blend,
        first_fold_idx=first_fold_idx,
    )
    # Whenever hyperopt submissions are trained outside of hyperopt, we need to update the summary.
    update_hyperopt_summary(
        base_predictors=base_predictors,
        ramp_kit_dir=str(ramp_kit_dir),
    )
    # Blend then bag the final blend of the hyperopt race on all the folds
    final_blend_then_bag(
        submissions=list(orchestrator.blended_submissions),
        ramp_kit_dir=str(ramp_kit_dir),
        kit_suffix=kit_suffix,
        n_folds_final_blend=n_folds_final_blend,
        first_fold_idx=first_fold_idx,
        round_datetime=orchestrator.final_round_datetime,
    )
    # Bag then blend the final blend of the hyperopt race on all the folds
    final_bag_then_blend(
        submissions=list(orchestrator.blended_submissions),
        ramp_kit_dir=str(ramp_kit_dir),
        kit_suffix=kit_suffix,
        n_folds_final_blend=n_folds_final_blend,
        first_fold_idx=first_fold_idx,
        round_datetime=orchestrator.final_round_datetime,
    )
    # Submit the best of each base submission (classical hyperopt)
    submit_best_submissions(
        base_predictors=base_predictors,
        ramp_kit_dir=str(ramp_kit_dir),
        kit_suffix=kit_suffix,
        n_folds_final_blend=n_folds_final_blend,
        first_fold_idx=first_fold_idx,
    )
