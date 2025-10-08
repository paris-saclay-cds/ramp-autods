import os
import glob
import pytest
import shutil
import hashlib
import numpy as np
from pathlib import Path

import rampds as rs
from rampds.fe_utils.utils import OPENFE_TEST_DIR

PATH = os.path.dirname(__file__)


def _generate_grid_path_kits():
    grid = []
    for path_kit in sorted(glob.glob(os.path.join(PATH, 'ramp_setup_kits', '*'))):
        grid.append(Path(path_kit).name)
    return grid


#TODO: try to understand whhy doesn't work on failure, kaggle failure and horses --> but seems to work on the nas kaggle failure
@pytest.mark.parametrize(
    "ramp_kit",
    _generate_grid_path_kits()
)
def test_submission(ramp_kit):
    kit_root = Path(PATH) / 'ramp_kits'
    setup_root = Path(PATH) / 'ramp_setup_kits'
    ramp_kit_dir = kit_root / f"{ramp_kit}_v0_n0"
    version_name = "0"
    number = 0
    feature_engineering = "openfe_test"

    # cleaning up
    if ramp_kit_dir.is_dir():
        shutil.rmtree(ramp_kit_dir)
    # TODO: add that back later
    # if Path('cache').is_dir():
    #     shutil.rmtree('cache')
    # if Path('catboost_info').is_dir():
    #     shutil.rmtree('catboost_info')
    # if Path(OPENFE_TEST_DIR).is_dir():
    #     shutil.rmtree(OPENFE_TEST_DIR)
    
#    ramp_kit_dir.mkdir(parents=True, exist_ok=True)

    # rs.scripts.setup.setup(
    #     ramp_kit = ramp_kit,
    #     setup_root = setup_root,
    #     kit_root = kit_root,
    #     version = "0",
    #     number = 0,
    #     # TODO: add an option w test that deletes the whole openfe_test directory after
    #     feature_engineering = "openfe_test",
    # )

    rs.setup(
        ramp_kit=ramp_kit,
        setup_root=setup_root,
        kit_root=kit_root,
        version=version_name,          
        number=number,
        feature_engineering=feature_engineering,
    )

    # cleaning up
    # if ramp_kit_dir.is_dir():
    #     shutil.rmtree(ramp_kit_dir)
    # if Path('cache').is_dir():
    #     shutil.rmtree('cache')
    # if Path('catboost_info').is_dir():
    #     shutil.rmtree('catboost_info')
    # # add deletion for openfe test directory
    # if Path(OPENFE_TEST_DIR).is_dir():
    #     shutil.rmtree(OPENFE_TEST_DIR)
