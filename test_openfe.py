import rampds
from rampds.fe_utils.utils import load_ramp_setup_kit_data, get_kaggle_ramp_setup_kit_path
from rampds.fe_utils.openfe_utils import *
from rampds.fe_utils.training import *
from rampds.scripts.openfe import OpenFEFeatureEngineering

data_name = 'wine'
ramp_setup_kit_path = get_kaggle_ramp_setup_kit_path(data_name)
train_df, test_df, metadata, _ = load_ramp_setup_kit_data(ramp_setup_kit_path)

openfe = OpenFEFeatureEngineering(
    train_df, 
    test_df, 
    metadata,
    data_name=data_name,
    n_cv_folds=3
    )

openfe.run_feature_engineering_and_selection()