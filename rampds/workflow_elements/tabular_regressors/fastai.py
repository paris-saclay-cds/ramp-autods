import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from fastai.tabular.all import *
from sklearn.base import BaseEstimator
from ramphy import Hyperparameter

# RAMP START HYPERPARAMETERS
bs = Hyperparameter(dtype='int', default=256, values=[128, 256, 512])
n_epochs = Hyperparameter(dtype='int', default=3, values=[3, 5, 10])#, 20, 30, 50])
learning_rate = Hyperparameter(dtype='float', default=1e-3, values=[3e-4, 1e-3, 3e-3])#, 1e-2, 3e-2])
wd = Hyperparameter(dtype='float', default=1e-2, values=[1e-4, 1e-3, 1e-2, 1e-1, 0.2, 0.3, 0.5])
ps = Hyperparameter(dtype='float', default=0.3, values=[0.0, 0.1, 0.3, 0.5])
layers = Hyperparameter(dtype='str', default='500-200', values=['200', '500', '1000', '200-100', '500-200', '1000-500', '200-100-50', '1000-500-200'])
use_bn = Hyperparameter(dtype='bool', default=True, values=[True, False])
train_bn = Hyperparameter(dtype='bool', default=True, values=[True, False])
moms = Hyperparameter(dtype='str', default='[0.95, 0.85, 0.95]', values=['[0.9, 0.9, 0.9]', '[0.95, 0.85, 0.95]', '[0.99, 0.9, 0.99]'])
one_cycle = Hyperparameter(dtype='bool', default=True, values=[True, False])
# RAMP END HYPERPARAMETERS

# Convert hyperparameters to the appropriate types
BS = int(bs)
N_EPOCHS = int(n_epochs)
LEARNING_RATE = float(learning_rate)
WD = float(wd)
PS = float(ps)
LAYERS = [int(x) for x in str(layers).split('-')]  # Convert string format like '500-200' to list [500, 200]
USE_BN = bool(use_bn)
TRAIN_BN = bool(train_bn)
MOMS = eval(str(moms))  # Convert string representation to actual list
ONE_CYCLE = bool(one_cycle)


# Custom tabular model that handles continuous-only data
class ContinuousTabularModel(Module):
    """Custom tabular model for continuous-only features"""
    def __init__(self, n_cont, out_sz, layers, ps=0.2, use_bn=True):
        super().__init__()
        # For continuous-only data, we skip all embedding code
        
        # Create network directly
        sizes = [n_cont] + layers + [out_sz]
        actns = [nn.ReLU(inplace=True) for _ in range(len(sizes)-2)]
        
        # For regression, don't add activation to final layer
        actns.append(None)
        
        # Build sequential layers
        layer_list = []
        for i, (in_sz, out_sz, actn) in enumerate(zip(sizes[:-1], sizes[1:], actns)):
            layer_list.append(nn.Linear(in_sz, out_sz))
            if use_bn and i < len(actns)-1: 
                layer_list.append(nn.BatchNorm1d(out_sz))
            if actn is not None: 
                layer_list.append(actn)
            if i < len(actns)-1: 
                layer_list.append(nn.Dropout(ps))
                
        self.layers = nn.Sequential(*layer_list)
    
    def forward(self, x_cat, x_cont):
        # For continuous-only data, ignore x_cat and use only x_cont
        return self.layers(x_cont)


class Regressor(BaseEstimator):
    def __init__(self, metadata):
        self.metadata = metadata
        self.target_cols = metadata["data_description"]["target_cols"]
        
        # Handle different types of regression metrics similar to the LightGBM example
        score_name = metadata.get("score_name", "mse")
        if score_name in ["mse", "rmse", "rmsle", "r2", "ngini"]:
            self.objective = "mse"
            self.loss_func = MSELossFlat()
            self.metrics = rmse
        elif score_name in ["mae", "medae", "smape"]:
            self.objective = "mae"
            self.loss_func = L1LossFlat()
            self.metrics = mae
        elif score_name in ["mare", "mape"]:
            self.objective = "mape"
            self.loss_func = MSELossFlat()  # FastAI doesn't have MAPE loss built-in, use MSE as default
            self.metrics = mape
        else:
            self.objective = "mse"
            self.loss_func = MSELossFlat()
            self.metrics = rmse
            
        # Check for multi-output regression
        self.multi_output = len(self.target_cols) > 1
                
        # Store procs for preprocessing
        self.procs = [Normalize] 
        
        # Internal tracking
        self.model = None
        self.learn = None
        self.dls = None
        
        # Log transform for RMSLE
        self.log_transform = (score_name == "rmsle")

    def _prepare_data(self, X, y=None):
        """Prepare data for fastai TabularPandas"""
        try:
            # If X is already a DataFrame, use it directly
            if isinstance(X, pd.DataFrame):
                df = X.copy()
            else:
                # Otherwise, try to convert it to a DataFrame
                df = pd.DataFrame(X)
                
            # Add target if provided (only for training)
            if y is not None:
                # For multiple target columns
                if self.multi_output:
                    if isinstance(y, pd.DataFrame):
                        for col in self.target_cols:
                            df[col] = y[col].values
                    else:
                        for i, col in enumerate(self.target_cols):
                            df[col] = y[:, i]
                else:
                    # Single target column (like in the classifier)
                    target_col = self.target_cols[0]
                    # Apply log transform if using RMSLE
                    if self.log_transform and y is not None:
                        df[target_col] = np.log1p(y)
                    else:
                        df[target_col] = y
                
            return df
        except Exception as e:
            print(f"Error preparing data: {{e}}")
            raise

    def fit(self, X, y):
        """Train the fastai tabular model for regression"""
        try:
            cont_names = list(X.columns)
            
            # Prepare training data
            df = self._prepare_data(X, y)
            
            # Handle multi-output regression
            if self.multi_output:
                # For multi-output, we'll use a regression block for each target
                to = TabularPandas(df, 
                                  procs=self.procs,
                                  cat_names=[],  # Empty for continuous-only data
                                  cont_names=cont_names,
                                  y_names=self.target_cols,
                                  y_block=RegressionBlock())
            else:
                # Single output regression
                to = TabularPandas(df, 
                                  procs=self.procs,
                                  cat_names=[],  # Empty for continuous-only data
                                  cont_names=cont_names,
                                  y_names=self.target_cols[0],
                                  y_block=RegressionBlock())
            
            # Create DataLoaders with proper batch size
            self.dls = to.dataloaders(bs=BS)
            
            # Create custom model for continuous-only data with output size matching targets
            n_out = len(self.target_cols) if self.multi_output else 1
            
            model = ContinuousTabularModel(
                n_cont=len(cont_names),
                out_sz=n_out,
                layers=LAYERS,
                ps=PS,
                use_bn=USE_BN
            )
            
            # Set up the learner with our custom model and appropriate loss function
            self.learn = Learner(
                self.dls,
                model,
                loss_func=self.loss_func,
                opt_func=Adam,
                metrics=self.metrics
            )
            
            # Set training status of BatchNorm layers
            if not TRAIN_BN:
                self.learn.model.apply(set_bn_eval)
            
            # Use mixed precision for faster training if available
            cbs = []
            if torch.cuda.is_available():
                cbs.append(MixedPrecision())
                            
            # Train model
            if ONE_CYCLE:
                self.learn.fit_one_cycle(N_EPOCHS, LEARNING_RATE, wd=WD, cbs=cbs)
            else:
                self.learn.fit(N_EPOCHS, LEARNING_RATE, wd=WD, cbs=cbs)
                
            # Store model
            self.model = self.learn.model
            
            return self
        except Exception as e:
            import traceback
            print(f"Error during model training: {{e}}")
            print(traceback.format_exc())
            raise

    def predict(self, X):
        """Get predictions for regression"""
        try:
            # Make sure model exists
            if self.model is None or self.learn is None:
                raise ValueError("Model has not been trained yet. Call fit() first.")
            
            # Prepare test data
            df = self._prepare_data(X)
            
            # Create test dataloader
            dl = self.learn.dls.test_dl(df)
            
            # Get predictions (no need for softmax since it's regression)
            preds, _ = self.learn.get_preds(dl=dl)
            
            # Convert to numpy array
            preds_np = preds.numpy()
            
            # Inverse transform for RMSLE
            if self.log_transform:
                preds_np = np.expm1(preds_np)
            
            # Reshape for sklearn compatibility if needed
            if preds_np.shape[1] == 1 and not self.multi_output:
                preds_np = preds_np.reshape(-1)
                
            return preds_np
        except Exception as e:
            print(f"Error during prediction: {{e}}")
            raise
