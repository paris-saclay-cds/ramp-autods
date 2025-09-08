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
        
        if out_sz > 1:  # For classification, don't add activation to final layer (handled by loss)
            actns.append(None)
        else:  # For regression, don't add activation to final layer
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


class Classifier(BaseEstimator):
    def __init__(self, metadata):
        self.metadata = metadata
        self.target_cols = metadata["data_description"]["target_cols"]
        if len(self.target_cols) > 1:
            raise NotImplementedError("Multi-output classification is not yet supported.")
        
        self.target_col = self.target_cols[0]
        target_value_dict = metadata["data_description"]["target_values"]
        self.target_values = target_value_dict[self.target_col]
        
        # Determine if binary or multi-class problem
        if len(self.target_values) == 2:
            self.is_binary = True
        else:
            self.is_binary = False
                
        # Store procs for preprocessing
        self.procs = [Normalize] 
        
        # Internal tracking
        self.model = None
        self.learn = None
        self.dls = None

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
                df[self.target_col] = y
                
            return df
        except Exception as e:
            print(f"Error preparing data: {{e}}")
            raise

    def fit(self, X, y):
        """Train the fastai tabular model"""
        try:
            cont_names=list(X.columns)
            # Prepare training data
            df = self._prepare_data(X, y)
            
            # Create TabularPandas object - important: set cat_names=[] to avoid the error
            to = TabularPandas(df, 
                              procs=self.procs,
                              cat_names=[],  # Empty for continuous-only data
                              cont_names=cont_names,
                              y_names=self.target_col,
                              y_block=CategoryBlock() if len(self.target_values) > 1 else RegressionBlock())
            
            # Create DataLoaders with proper batch size
            self.dls = to.dataloaders(bs=BS)
            
            # Get the number of classes for output size
            n_out = len(self.target_values) if len(self.target_values) > 1 else 1
            
            # Create custom model for continuous-only data
            model = ContinuousTabularModel(
                n_cont=len(cont_names),
                out_sz=n_out,
                layers=LAYERS,
                ps=PS,
                use_bn=USE_BN
            )
            
            # For classification, use softmax output
            if len(self.target_values) > 1:
                loss_func = CrossEntropyLossFlat()
            else:
                loss_func = MSELossFlat()
            
            # Set up the learner with our custom model
            self.learn = Learner(
                self.dls,
                model,
                loss_func=loss_func,
                opt_func=Adam,
                metrics=accuracy if len(self.target_values) > 1 else rmse
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

    def predict_proba(self, X):
        """Get probability predictions"""
        try:
            # Make sure model exists
            if self.model is None or self.learn is None:
                raise ValueError("Model has not been trained yet. Call fit() first.")
            
            # Prepare test data
            df = self._prepare_data(X)
            
            # Create test dataloader
            dl = self.learn.dls.test_dl(df)
            
            # Get predictions with no_loss=True to ensure softmax is applied
            preds, _ = self.learn.get_preds(dl=dl, with_decoded=False, with_loss=False)
            
            # If binary/multi-class, ensure we have probabilities by applying softmax if needed
            if len(self.target_values) > 1 and not (preds.min() >= 0 and preds.max() <= 1):
                preds = F.softmax(preds, dim=1)
            
            # Convert to numpy array
            preds_np = preds.numpy()
            
            # For binary classification, ensure proper format (sklearn expects [neg_prob, pos_prob])
            if self.is_binary:
                # Check if we need to swap columns based on class encoding
                if len(preds_np.shape) == 1 or preds_np.shape[1] == 1:
                    # Handle case where model outputs single probability
                    pos_probs = preds_np.reshape(-1)
                    neg_probs = 1 - pos_probs
                    return np.column_stack((neg_probs, pos_probs))
                else:
                    # Already have both probabilities
                    return preds_np
            else:
                # For multiclass, return probability for each class
                return preds_np
        except Exception as e:
            print(f"Error during prediction: {{e}}")
            raise