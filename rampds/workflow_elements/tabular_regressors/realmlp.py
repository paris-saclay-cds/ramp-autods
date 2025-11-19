import numpy as np
from sklearn.base import BaseEstimator
from ramphy import Hyperparameter
from pytabkit import RealMLP_TD_Regressor

# RAMP START HYPERPARAMETERS
n_layers = Hyperparameter(dtype='int', default=1, values=[1, 2, 3, 4, 5, 6, ])
p_drop = Hyperparameter(dtype='float', default=0.3, values=[0.0, 0.1, 0.2, 0.3, 0.4])
layer_size = Hyperparameter(dtype='int', default=64, values=[32, 64, 128, 256, 512])
learning_rate = Hyperparameter(dtype='float', default=0.1, values=[0.02, 0.05, 0.1, 0.3])
plr_sigma = Hyperparameter(dtype='float', default=1.0, values=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0])
sq_mom = Hyperparameter(dtype='float', default=0.995, values=[0.95, 0.97, 0.99, 0.995])
plr_lr_factor = Hyperparameter(dtype='float', default=0.1, values=[0.05, 0.07, 0.1, 0.2, 0.3])
scale_lr_factor = Hyperparameter(dtype='float', default=7.0, values=[2.0, 3.0, 5.0, 7.0, 10.0])
first_layer_lr_factor = Hyperparameter(dtype='float', default=0.4, values=[0.3, 0.4, 0.7, 1.0, 1.5])
ls_eps = Hyperparameter(dtype='float', default=0.005, values=[0.005, 0.007, 0.01, 0.05, 0.1])
wd = Hyperparameter(dtype='float', default=0.01, values=[0.0, 0.01, 0.03, 0.05, 0.1, 0.3])
# RAMP END HYPERPARAMETERS

#params = {{
#        "use_ls": rng.choice(
#            ["auto", True]
#        ),  # use label smoothing (will be ignored for regression)
#    }}

N_LAYERS = int(n_layers)
LAYER_SIZE = int(layer_size)
P_DROP = float(p_drop)
N_EPOCHS = 256
LEARNING_RATE = float(learning_rate)
PLR_SIGMA = float(plr_sigma)
SQ_MOM = float(sq_mom)
PLR_LR_FACTOR = float(plr_lr_factor)
SCALE_LR_FACTOR = float(scale_lr_factor)
FIRST_LAYER_LR_FACTOR = float(first_layer_lr_factor)
LS_EPS = float(ls_eps)
WD = float(wd)

class Regressor(BaseEstimator):
    def __init__(self, metadata):
        self.metadata = metadata
        score_name = metadata["score_name"]
        if score_name in ["mse", "rmse", "rmsle", "r2", "ngini"]:
            self.objective = "rmse"
        elif score_name in ["mae", "medae", "smape", "mare", "mape"]:
            self.objective = "mae"
        else:
            raise ValueError(f"Unknown score_name {score_name}")

    def fit(self, X, y):
        if self.metadata["score_name"] == "rmsle":
            y = np.log1p(y)
        batch_size = 2048
        self.reg = RealMLP_TD_Regressor(
            n_epochs=N_EPOCHS,
            batch_size=batch_size,
            hidden_sizes=[LAYER_SIZE] * N_LAYERS,
            p_drop=P_DROP,
            lr=LEARNING_RATE,
            plr_sigma=PLR_SIGMA,
            sq_mom=SQ_MOM,
            plr_lr_factor=PLR_LR_FACTOR,
            scale_lr_factor=SCALE_LR_FACTOR,
            first_layer_lr_factor=FIRST_LAYER_LR_FACTOR,
            ls_eps=LS_EPS,
            wd=WD,
            val_metric_name=self.objective,
            use_early_stopping=False,
            act="mish",
            p_drop_sched="flat_cos",
            ls_eps_sched="coslog4",
            verbosity=2
        )
        self.reg.fit(X, y)

    def predict(self, X):
        y_pred = self.reg.predict(X)
        if self.metadata["score_name"] == "rmsle":
            y_pred = np.expm1(y_pred)
        return y_pred
