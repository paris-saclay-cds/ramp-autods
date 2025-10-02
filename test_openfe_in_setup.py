import rampds as rs
from rampds.new_utils.utils import RAMP_SETUP_KITS_NAS_PATH, RAMP_KITS_NAS_PATH

# ramp_kit = "kaggle_abalone"
ramp_kit = "kaggle_wine"
setup_root = RAMP_SETUP_KITS_NAS_PATH
kit_root = "openfe_new_setup/"
version = "OpenFE_test"
number = 0

# Use the original setup function
rs.setup(
    ramp_kit=ramp_kit,
    setup_root=setup_root,
    kit_root=kit_root,
    version=version,          
    number=number,
    openfe_feature_engineering=True,
)