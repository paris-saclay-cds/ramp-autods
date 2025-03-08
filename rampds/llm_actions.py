import json
import subprocess
from pathlib import Path
from typing import Callable, Dict, Optional

import ramphy as rh
import rampds as rs


@rh.actions.ramp_action
def submit_llm_feature_rejector(
    asking_function: Callable,
    parser_function: Callable,
    pronpt_builder: Callable,
    ramp_kit_dir: Path | str,
    submission: Path | str,
    ramp_data_dir: Optional[str | Path] = None,
):
    """This function asks the llm for a feature to drop and submits a preprocessor that does that

    Args:
        asking_function (Callable): Function that uses the LLM to generate and answer
        parser_function (Callable): Function used to parse the LLM answer. It has to return the name of the feat to drop
        pronpt_builder (Callable): Function to generate the prompt from the prompt template
        ramp_kit_dir (Path | str): Path of the ramp kit dir
        submission (Path | str): Name of the submission
        ramp_data_dir (Optional[str  |  Path], optional): Path of the ramp data dir. Defaults to None.
    """
    ramp_kit_dir, ramp_data_dir = rh.actions.convert_ramp_dirs(ramp_kit_dir, ramp_data_dir)
    (ramp_kit_dir / "submissions" / submission).mkdir(parents=True, exist_ok=True)
    metadata = json.load(open(ramp_data_dir / "data" / "metadata.json"))

    # Ask the LLM which feature to drop
    # TODO prepare prompt
    # TODO need task description and challenge description in the metadata
    # Also need a view of the dataset (maybe df.head)
    prompt = pronpt_builder(["prompts/drop_feature.jinja"], {"metadata": metadata})
    feature_to_drop = parser_function(asking_function(prompt))

    dp_idx = rs.utils.num_data_preprocessors(submission, ramp_kit_dir)

    # Load and prepare the preprocessor from the template
    dp_template_path = Path("workflow_elements") / "tabular_data_preprocessors" / "llm_drop_column.py"
    dp_code = rs.utils.load_template(package=rs, template_path=dp_template_path)
    dp_code = dp_code.format(dropped_feat=feature_to_drop)

    # Add the preprocessor in the submission folder
    with open(
        ramp_kit_dir / "submissions" / submission / f"data_preprocessor_{dp_idx}_llm_drop_{feature_to_drop}.py", "w"
    ) as f_out:
        f_out.write(dp_code)


def execute_script(script_path: str | Path, env_args: Dict, script_args: Dict, hydra_args: Dict) -> bool:
    """executes the required python script

    Args:
        script_path (str | Path): _description_
        env_args (Dict): _description_
        script_args (Dict): _description_
        hydra_args (Dict): _description_

    Returns:
        bool: True if terminated properly, False otherwise
    """
    # Ensure script_path is a Path object
    script_path = str(script_path)

    # Add environment args
    cmd = []
    for key, value in env_args.items():
        cmd.append(f"{key}={value} ")

    # Construct the command to run the script with its arguments
    cmd = cmd + ["python", script_path]
    for key, value in script_args.items():
        cmd.append(f"--{key}")
        cmd.append(str(value))

    # Add any arguments for hydra (these are handled differently than script)
    for key, value in hydra_args.items():
        cmd.append(f"{key}={value} ")

    try:
        # Run the script
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        # Optionally, you can print or log the output
        print(result.stdout)
        print(result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        # Handle errors in the script execution
        print(f"Script failed with exit code {e.returncode}")
        print(e.output)
        return False
    except Exception as e:
        # Handle other potential exceptions
        print(f"An error occurred: {e}")
        return False
