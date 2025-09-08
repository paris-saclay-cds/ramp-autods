import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
#from mbrltools.pytorch_utils import train
from ramphy import Hyperparameter
#from rampwf.utils import BaseGenerativeRegressor
from sklearn.utils import check_random_state

#torch.manual_seed(7)

# RAMP START HYPERPARAMETERS
nn_type = Hyperparameter(dtype='str', default='NN', values=['NN', 'MDN'])
n_layers = Hyperparameter(dtype='int', default=4, values=[1, 2, 3, 4, 5, 6])
layer_size = Hyperparameter(dtype='int', default=32, values=[32, 64, 128, 256, 512])
drop_first = Hyperparameter(dtype='float', default=0.0, values=[0.0, 0.1, 0.2])
drop_repeated = Hyperparameter(dtype='float', default=0.0, values=[0.0, 0.1, 0.2])
y_scale = Hyperparameter(dtype='float', default=1.0, values=[0.0, 1.0, 10.0, 100.0])
n_gaussians = Hyperparameter(dtype='int', default=1, values=[1, 2, 3, 4, 5, 7, 10])
sigma_transf = Hyperparameter(dtype='str', default='asym_sigmoid', values=['asym_sigmoid', 'exp'])
# RAMP END HYPERPARAMETERS

is_diff = False
first_n_steps = 10000000000
last_n_steps = 10000000000
sampling = "det"
n_epochs = 200
learning_rate = 0.001


VALIDATION_FRACTION = 0.1

# rampwf generative_regression START
import inspect
import os
import json

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_random_state

# The maximum numbers of parameters a distribution would need
# Only matters for bagging mixture models in
# prediction_types.generative_regression
MAX_MIXTURE_PARAMS = 6
EMPTY_DIST = -1

# We explcitly enumerate all scipy distributions here so their codes
# do not change even if scipy adds new distributions. We considered but
# discarded using a hash code.
distributions_dict = {{
    'norm': 0,
    'uniform': 1,
    'beta': 2,
    'truncnorm': 3,
    'foldnorm': 4,
    'vonmises': 5,
    'ksone': 6,
    'kstwo': 7,
    'kstwobign': 8,
    'alpha': 9,
    'anglit': 10,
    'arcsine': 11,
    'betaprime': 12,
    'bradford': 13,
    'burr': 14,
    'burr12': 15,
    'fisk': 16,
    'cauchy': 17,
    'chi': 18,
    'chi2': 19,
    'cosine': 20,
    'dgamma': 21,
    'dweibull': 22,
    'expon': 23,
    'exponnorm': 24,
    'exponweib': 25,
    'exponpow': 26,
    'fatiguelife': 27,
    'foldcauchy': 28,
    'f': 29,
    'weibull_min': 30,
    'weibull_max': 31,
    'frechet_r': 32,
    'frechet_l': 33,
    'genlogistic': 34,
    'genpareto': 35,
    'genexpon': 36,
    'genextreme': 37,
    'gamma': 38,
    'erlang': 39,
    'gengamma': 40,
    'genhalflogistic': 41,
    'gompertz': 42,
    'gumbel_r': 43,
    'gumbel_l': 44,
    'halfcauchy': 45,
    'halflogistic': 46,
    'halfnorm': 47,
    'hypsecant': 48,
    'gausshyper': 49,
    'invgamma': 50,
    'invgauss': 51,
    'geninvgauss': 52,
    'norminvgauss': 53,
    'invweibull': 54,
    'johnsonsb': 55,
    'johnsonsu': 56,
    'laplace': 57,
    'levy': 58,
    'levy_l': 59,
    'levy_stable': 60,
    'logistic': 61,
    'loggamma': 62,
    'loglaplace': 63,
    'lognorm': 64,
    'gilbrat': 65,
    'maxwell': 66,
    'mielke': 67,
    'kappa4': 68,
    'kappa3': 69,
    'moyal': 70,
    'nakagami': 71,
    'ncx2': 72,
    'ncf': 73,
    't': 74,
    'nct': 75,
    'pareto': 76,
    'lomax': 77,
    'pearson3': 78,
    'powerlaw': 79,
    'powerlognorm': 80,
    'powernorm': 81,
    'rdist': 82,
    'rayleigh': 83,
    'loguniform': 84,
    'reciprocal': 85,
    'rice': 86,
    'recipinvgauss': 87,
    'semicircular': 88,
    'skewnorm': 89,
    'trapz': 90,
    'triang': 91,
    'truncexpon': 92,
    'tukeylambda': 93,
    'vonmises_line': 94,
    'wald': 95,
    'wrapcauchy': 96,
    'gennorm': 97,
    'halfgennorm': 98,
    'crystalball': 99,
    'argus': 100,
    'binom': 101,
    'bernoulli': 102,
    'betabinom': 103,
    'nbinom': 104,
    'geom': 105,
    'hypergeom': 106,
    'logser': 107,
    'poisson': 108,
    'planck': 109,
    'boltzmann': 110,
    'randint': 111,
    'zipf': 112,
    'dlaplace': 113,
    'skellam': 114,
    'yulesimon': 115
}}

_inverted_scipy_dist_dict = dict(map(reversed, distributions_dict.items()))


def distributions_dispatcher(d_type=-1):
    try:
        name = _inverted_scipy_dist_dict[d_type]
    except KeyError:
        raise KeyError("%s not a valid distribution type." % d_type)
    return getattr(stats, name)


def get_n_params(dist):
    return len(inspect.signature(dist._parse_args).parameters)


class MixtureYPred:
    """
    Object made to convert outputs of generative regressors to a numpy array
    representation (y_pred) used in RAMP.
    Works for autoregressive and independent, not the full case.
    """

    def __init__(self):
        self.dims = []

    def add(self, weights, types, params):
        """
        Must be called every time we get a prediction, creates the
        distribution list

        Parameters
        ----------
        weights : numpy array (n_timesteps, n_component_per_dim)
            the weights of the mixture for current dim

        types : numpy array (n_timesteps, n_component_per_dim)
            the types of the mixture for current dim

        params : numpy array (n_timesteps,
                              n_component_per_dim*n_param_per_dist)
            the params of the mixture for current dim, the order must
            correspond to the one of types
        """
        n_components_curr = types.shape[1]
        sizes = np.full((len(types), 1), n_components_curr)
        result = np.concatenate(
            (sizes, weights, types, params), axis=1)
        self.dims.append(result)
        return self

    def finalize(self, order=None):
        """
        Must called be once all the dims were added

        Parameters
        ----------
        order : list
            The order in which the dims should be sorted
        """
        dims_original_order = np.array(self.dims)
        if order is not None:
            dims_original_order = dims_original_order[np.argsort(order)]
        return np.concatenate(dims_original_order, axis=1)


def get_components(curr_idx, y_pred):
    """Extracts dimensions from the whole y_pred array.

    These dimensions can then be used elsewhere (e.g. to compute the pdf).
    It is meant to be called like so:

    curr_idx=0
    for dim in dims:
        curr_idx, ... = get_components(curr_idx, y_pred)

    Parameters
    ----------
    curr_idx : int
        The current index in the whole y_pred.
    y_pred : numpy array
        An array built using MixtureYPred "add" and "finalize".

    Return
    ------
    curr_idx : int
        The current index in the whole y_pred after recovering the current
        dimension
    n_components : int
        The number of components in the mixture for the current dim
    weights : numpy array (n_timesteps, n_component_per_dim)
        The weights of the mixture for current dim
    types : numpy array (n_timesteps, n_component_per_dim)
        The types of the mixture for current dim
    dists : list of objects extending AbstractDists
        A list of distributions to be used for current dim
    paramss : numpy array (n_timesteps, n_component_per_dim*n_param_per_dist)
        The params of the mixture for current dim, that align with the
        other returned values
    """
    n_components = int(y_pred[0, curr_idx])
    curr_idx += 1
    id_params_start = curr_idx + n_components * 2
    weights = y_pred[:, curr_idx:curr_idx + n_components]
    assert (weights >= 0).all(), "Weights should all be positive."
    weights /= weights.sum(axis=1)[:, np.newaxis]
    types = y_pred[:, curr_idx + n_components:id_params_start]
    curr_idx = id_params_start
    dists = []
    paramss = []
    for i in range(n_components):
        non_empty_mask = ~np.array(types[:, i] == EMPTY_DIST)
        curr_types = types[:, i][non_empty_mask]
        curr_type = curr_types[0]
        assert np.all(curr_type == curr_types)  # component types must be fixed
        dists.append(distributions_dispatcher(curr_type))
        end_params = curr_idx + get_n_params(dists[i])
        paramss.append(y_pred[:, curr_idx:end_params])
        curr_idx = end_params
    return curr_idx, n_components, weights, types, dists, paramss


class BaseGenerativeRegressor(BaseEstimator):
    """Base class for generative regressors.

    Provides a sample method for generative regressors which return an explicit
    density (they have a predict method).
    Provides a predict method for generative regressors which do not have an
    explicity density but can be sampled from easily (they have a sample
    method).

    Parameters
    ----------
    decomposition : None or string
        Decomposition of the joint distribution for multivariate outputs.
    """
    # number of samples used to estimate the conditional distribution of the
    # output given the input. we use a class attribute to not have to
    # to call super().__init__() in the submissions.
    n_samples = 30

    def __init__(self):
        # this method is here to be able to instantiate the class for testing
        # purpose
        self.decomposition = None

    def samples_to_distributions(self, samples):
        """Estimate output conditional distributions.

        For each timestep, estimate the conditional output distribution from
        the draws contained in the samples array. The distribution is
        estimated with a kernel density estimator thus returning a mixture.

        This method is useful for generative regressors that can only be
        sampled from but do not provide an explicit likelihood. Examples
        of such generative regressors are Variational Auto Encoders and flow
        based methods.

        Parameters
        ----------
        samples : numpy array of shape [n_timesteps, n_targets, n_samples]
            For each timestep, an array of samples sampled from a generative
            regressor.

        Return
        ------
        weights : numpy array of float
            discrete probabilities of each component of the mixture
        types : list of strings
            scipy names referring to component of the mixture types.
            see https://docs.scipy.org/doc/scipy/reference/stats.html.
            In this case, they are normal
        params : numpy array
            Parameters for each component in the mixture. mus are the given
            samples, sigmas are estimated using silverman method.
        """
        n_timesteps, n_targets, n_samples = samples.shape
        mus = samples
        weights = np.full((n_timesteps, n_targets * n_samples), 1 / n_samples)
        sigmas = np.empty((n_timesteps, n_targets, n_samples))
        for i in range(n_timesteps):
            kde = stats.gaussian_kde(samples[i, ...], bw_method='silverman')
            bandwidths = np.sqrt(np.diag(kde.covariance)).reshape(-1, 1)
            sigmas[i, ...] = np.repeat(bandwidths, n_samples, axis=1)

        params = np.empty((n_timesteps, mus.shape[1] * mus.shape[2] * 2))
        params[:, 0::2] = mus.reshape(n_timesteps, -1)
        params[:, 1::2] = sigmas.reshape(n_timesteps, -1)
        types = ['norm'] * n_samples * mus.shape[1]

        return weights, types, params

    def _sample(self, distribution, rng):
        """Draw one sample from the input distribution.

        The distribution is assumed to be a mixture (a gaussian mixture if
        decomposition is set to None).

        Parameters
        ----------
        distribution : tuple of numpy arrays (weights, types, parameters)
            A mixture distribution characterized by weights, distribution types
            and associated distribution parameters.

        rng : Random state object
            The RNG or the state of the RNG to be used when sampling.

        Returns
        -------
        y_sampled : numpy array, shape (1, n_targets) if decomposition is not
        None, shape (n_samples, n_targets) if decomposition is None
            The sampled targets. n_targets is equal to 1 if decomposition is
            not None.
        """
        if self.decomposition is None:
            weights, types, params = distribution
            # the weights are all the same for each dimension: we keep only
            # the ones of the first dimension, the final shape is
            # (n_samples, n_components)
            n_samples = weights.shape[0]
            weights = weights.reshape(self._n_targets, n_samples, -1)[0]

            # we convert the params mus and sigmas back to their shape
            # (n_samples, n_targets, n_components) as it is then easier to
            # retrieve the ones that we need.
            all_mus = params[:, 0::2].reshape(n_samples, self._n_targets, -1)
            all_sigmas = params[:, 1::2].reshape(
                n_samples, self._n_targets, -1)

            # sample from the gaussian mixture
            weights /= np.sum(weights, axis=1)[:, np.newaxis]
            # vectorize sampling of one component for each sample
            cum_weights = weights.cumsum(axis=1)
            sampled_components = (
                (cum_weights > rng.rand(n_samples)[:, np.newaxis])
                .argmax(axis=1))
            # get associated means and sigmas
            all_ind = np.arange(n_samples)
            sampled_means = all_mus[all_ind, :, sampled_components]
            sampled_sigmas = all_sigmas[all_ind, :, sampled_components]

            y_sampled = rng.randn(n_samples, self._n_targets) * sampled_sigmas
            y_sampled += sampled_means
        else:  # autoregressive or independent decomposition.
            weights, types, params = distribution

            if weights.shape[0] > 1:
                raise ValueError(
                    'You are trying to sample more than 1 sample without your '
                    'own sample method. Using the sample method inherited from'
                    ' BaseGenerativeRegressor is not supporting this. '
                    'Supporting it can be made with a simple (slow) for loop.')

            n_dists = len(types)
            try:
                types = [distributions_dict[type_name] for type_name in types]
            except KeyError:
                message = ('One of the type names is not a valid Scipy '
                           'distribution')
                raise AssertionError(message)
            types = np.array([types, ] * len(weights))

            w = weights[0].ravel()
            w = w / sum(w)
            selected = rng.choice(n_dists, p=w)
            dist = distributions_dispatcher(int(types[0, selected]))

            # find which params to take: this is needed if we have a
            # mixture of different distributions with different number of
            # parameters
            sel_id = 0
            for k in range(selected):
                curr_type = distributions_dispatcher(int(types[0, k]))
                sel_id += get_n_params(curr_type)
            y_sampled = dist.rvs(
                *params[0, sel_id:sel_id + get_n_params(dist)])
            y_sampled = np.array(y_sampled)
        return y_sampled

    def sample(self, X, rng=None, restart=None):
        """Draw a sample from the conditional output distribution given X.

        X is assumed to contain only one timestep. The conditional output
        distribution given X is predicted and a sample is drawn from this
        distribution.

        This method must be overriden for generative regressors that can be
        naturally sampled from such as Variational Auto Encoders.

        Parameters
        ----------
        X : numpy array, shape (1, n_features)
            Input timestep for which we want to sample the output from the
            conditional predicted distribution.
        rng : Random state object
            The RNG or the state of the RNG to be used when sampling.
        restart : string
            Name of the restart column. None is no restart.
        """
        rng = check_random_state(rng)

        if restart is not None:
            distribution = self.predict(X, restart)
        else:
            distribution = self.predict(X)

        return self._sample(distribution, rng)

    def predict(self, X, restart=None):
        """Predict conditional output distributions for each timestep in X.

        This method is to be used for generative regressors only providing a
        sample method. Samples are drawn with the sample method and
        distributions are estimated from these samples.

        This method should be overriden for generative regressors providing
        an explicity density.

        Parameters
        ----------
        X : numpy array, shape (n_timesteps, n_features)
            Input timesteps for which we want an estimated output distribution.
        restart : string
            Name of the restart column. None is no restart.
        """
        samples = []
        for _ in range(self.n_samples):
            if restart is not None:
                sampled = self.sample(X, restart)
            else:
                sampled = self.sample(X)
            samples.append(sampled)
        samples = np.stack(samples, axis=2)

        return self.samples_to_distributions(samples)


def _reorder_targets(module_path, y_array, target_column_names):
    """Find submitted order and reorder the targets."""
    order_path = os.path.join(module_path, 'order.json')
    try:
        with open(order_path, "r") as json_file:
            order = json.load(json_file)
            # Check if the names in the order and observables are all here
            if set(order.keys()) == set(target_column_names):
                # We sort the variable names by user-defined order
                order = [k for k, _ in sorted(
                    order.items(), key=lambda item: item[1])]
                # Map it to original order
                order = [target_column_names.index(i) for i in order]
                print(order)
                y_array = y_array[:, order]
            else:
                raise RuntimeError("Order variables are not correct")
    except FileNotFoundError:
        print("Using default order")
        order = range(len(target_column_names))
    return y_array, order
# rampwf generative_regression END


# mbrl-tools pytroch_train START
import copy
import csv
import io
import os
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import torch
import torch.distributions as D
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torch_data
from sklearn.utils import check_random_state
from torch.autograd import Variable
#from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
#import ot

class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, patience=5, min_delta=1e-2):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif (self.best_loss - val_loss) / self.best_loss >= self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif (self.best_loss - val_loss) / self.best_loss < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                print("\n INFO: Early stopping")
                self.early_stop = True
            else:
                print(f"Patience counter = {{self.counter}} < {{self.patience}}")
        return self.early_stop


class UnlabeledTensorDataset(torch_data.TensorDataset):
    """Dataset wrapping unlabeled data tensors.

    Each sample will be retrieved by indexing tensors along the first
    dimension.

    Arguments:
        data_tensor (Tensor): contains sample data.
    """

    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __getitem__(self, index):
        return self.data_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)


def _set_device(disable_cuda=False):
    """Set device to CPU or GPU.

    Parameters
    ----------
    disable_cuda : bool (default=False)
        Whether to use CPU instead of GPU.

    Returns
    -------
    device : torch.device object
        Device to use (CPU or GPU).
    """
    # XXX we might also want to use CUDA_VISIBLE_DEVICES if it is set
    if not disable_cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    return device

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math

class SuperVerboseReduceLROnPlateau(ReduceLROnPlateau):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_count = 0
        
    def step(self, metrics, epoch=None):
        self.step_count += 1
        print(f"\n{{'='*60}}")
        print(f"SCHEDULER STEP {{self.step_count}}")
#        print(f"{{'='*60}}")
        
        # Convert metrics to float if it's a tensor
        current = float(metrics)
        print(f"📊 Current metric: {{current:.6f}}")
        
        # Print current state
#        print(f"🎯 Mode: {{self.mode}}")
#        print(f"⏰ Patience: {{self.patience}}")
#        print(f"📉 Factor: {{self.factor}}")
#        print(f"🔢 Threshold: {{self.threshold}} ({{self.threshold_mode}})")
#        print(f"❄️  Cooldown: {{self.cooldown}}")
#        print(f"🔒 Min LR: {{self.min_lrs}}")
        
        # Print internal state before processing
        print(f"\n--- BEFORE PROCESSING ---")
        print(f"🏆 Best metric so far: {{self.best if hasattr(self, 'best') else 'None'}}")
        print(f"😞 Bad epochs count: {{self.num_bad_epochs}}")
        print(f"🧊 Cooldown counter: {{self.cooldown_counter}}")
        print(f"📚 Current LR: {{[group['lr'] for group in self.optimizer.param_groups]}}")
        
        # Check if we're in cooldown
        if self.cooldown_counter > 0:
            print(f"\n🧊 IN COOLDOWN! Counter: {{self.cooldown_counter}}")
            self.cooldown_counter -= 1
            print(f"🧊 Cooldown counter decremented to: {{self.cooldown_counter}}")
            return
            
        # Initialize best if first time
        if self.best is None:
            print(f"\n🆕 FIRST TIME - Initializing best to: {{current}}")
            self.best = current
            return
            
        # Check for improvement
        print(f"\n--- CHECKING FOR IMPROVEMENT ---")
        
        if self.mode == 'min':
            # For minimization (like loss)
            rel_epsilon = 1. - self.threshold
            abs_epsilon = -self.threshold
            is_better = current < self.best * rel_epsilon or current < self.best + abs_epsilon
            improvement = self.best - current
        else:
            # For maximization (like accuracy)  
            rel_epsilon = self.threshold + 1.
            abs_epsilon = self.threshold
            is_better = current > self.best * rel_epsilon or current > self.best - abs_epsilon
            improvement = current - self.best
            
#        print(f"🔍 Checking improvement...")
#        print(f"   Current: {{current:.6f}}")
#        print(f"   Best: {{self.best:.6f}}")
        print(f"   Improvement: {{improvement:.6f}}")
#        print(f"   Threshold mode: {{self.threshold_mode}}")
        
        if self.threshold_mode == 'rel':
            threshold_value = self.best * self.threshold if self.mode == 'min' else self.best * self.threshold
#            print(f"   Relative threshold: {{threshold_value:.6f}}")
        else:
#            print(f"   Absolute threshold: {{self.threshold:.6f}}")
            pass            
#        print(f"   Is better? {{is_better}}")
        
        if is_better:
            print(f"\n✅ IMPROVEMENT DETECTED!")
            print(f"   Old best: {{self.best:.6f}}")
            print(f"   New best: {{current:.6f}}")
#            print(f"   Resetting bad epochs counter from {{self.num_bad_epochs}} to 0")
            
            self.best = current
            self.num_bad_epochs = 0
        else:
            print(f"\n❌ NO IMPROVEMENT")
            self.num_bad_epochs += 1
            print(f"   Bad epochs incremented to: {{self.num_bad_epochs}}")
#            print(f"   Patience limit: {{self.patience}}")
            
            if self.num_bad_epochs > self.patience:
                print(f"\n🚨 PATIENCE EXCEEDED! ({{self.num_bad_epochs}} > {{self.patience}})")
                self._reduce_lr(epoch)
                self.cooldown_counter = self.cooldown
                self.num_bad_epochs = 0
                print(f"   Cooldown counter set to: {{self.cooldown_counter}}")
                print(f"   Bad epochs reset to: 0")
            else:
                remaining_patience = self.patience - self.num_bad_epochs
                print(f"   Patience remaining: {{remaining_patience}}")
                
        print(f"\n--- AFTER PROCESSING ---")
        print(f"🏆 Best metric: {{self.best:.6f}}")
        print(f"😞 Bad epochs: {{self.num_bad_epochs}}")
        print(f"🧊 Cooldown counter: {{self.cooldown_counter}}")
        print(f"📚 Final LR: {{[group['lr'] for group in self.optimizer.param_groups]}}")
        print(f"{{'='*60}}\n")

    def _reduce_lr(self, epoch):
        print(f"\n🔥 REDUCING LEARNING RATE!")
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            
            print(f"   Group {{i}}:")
            print(f"     Old LR: {{old_lr:.8f}}")
            print(f"     Factor: {{self.factor}}")
            print(f"     Calculated new LR: {{old_lr * self.factor:.8f}}")
            print(f"     Min LR: {{self.min_lrs[i]:.8f}}")
            print(f"     Final new LR: {{new_lr:.8f}}")
            
            if new_lr < old_lr:
                param_group['lr'] = new_lr
                if self.verbose:
                    print(f"     ✅ LR reduced: {{old_lr:.8f}} -> {{new_lr:.8f}}")
            else:
                print(f"     ⚠️  LR already at minimum!")

        
def train(
    model,
    dataset_train,
    dataset_valid=None,
    validation_fraction=None,
    n_epochs=10,
    batch_size=128,
    loss_fn=nn.MSELoss(),
    optimizer=None,
    scheduler=None,
    min_lr=0,
    return_best_model=False,
    disable_cuda=False,
    batch_size_predict=None,
    drop_last=False,
    numpy_random_state=None,
    is_vae=False,
    is_nvp=False,
    is_packed_autoreg=False,
    val_loss_fn=None,
    verbose=False,
    shuffle=True,
    tensorboard_path=None,
    log_loss=False,
    sampler=None,
    early_stopping=None,
    target_dim=None,
    clip_gradients=False,
    clip_gradients_value=200.0,
    keep_checkpoints=[],
):
    """Training model using the provided dataset and given loss function.

    model : pytorch nn.Module
        Model to be trained.

    dataset_train : Tensor dataset.
        Training data set.

    dataset_valid : Tensor dataset.
        If not None, data set used to compute a validation loss. This data set
        is not used to train the model.

    validation_fraction : float in (0, 1).
        If not None, fraction of samples from dataset to put aside to be
        use as a validation set. If dataset_valid is not None then
        dataset_valid overrides validation_fraction.

    n_epochs : int
        Number of epochs

    batch_size : int
        Batch size.

    loss_fn : function
        Pytorch loss function.

    optimizer : object
        Pytorch optimizer

    scheduler : object
        Pytorch scheduler.

    return_best_model : bool
        Whether to return the best model on the validation loss. More exactly,
        if set to True, the model trained at the epoch that lead to the best
        performance on the validation dataset is returned. In this case the
        best validation loss is also returned.

    disable_cuda : bool
        Whether to use CPU instead of GPU.

    batch_size_predict : int
        Batch size to use for the computation of the validation loss
        in case of a very large valid dataset. If None, no batch size is used.

    drop_last : bool
        Whether to drop the last batch in the dataloader if incomplete.

    numpy_random_state : int or numpy RNG
        Used when shuffling the training dataset before splitting it into
        a training and a validation datasets.

    is_vae : bool
        Whether the model we are training is a VAE.

    is_nvp : bool
        Whether the model we are training is a RealNVP.

    val_loss_fn : function
        The function to be used for valid loss.
        If None, train_loss will be used.

    verbose : bool
        Whether to print training information.

    shuffle : bool
        Whether to drop shuffle the data.

    tensorboard_path : string
        Path to the tensorboard directory. If set to none, ignored.


    early_stopping : object
        EarlyStopping object

    Returns
    -------
    model : pytorch nn.Module
        Trained model. If return_best_model is set to True the best validation
        loss is also returned.

    """
    # This makes the training extremely slow but useful for debugguing
    # torch.autograd.set_detect_anomaly(True)

    # use GPU by default if cuda is available, otherwise use CPU
    device = _set_device(disable_cuda=disable_cuda)
    model = model.to(device)
    val_loss = 0
    numpy_rng = check_random_state(numpy_random_state)

    if optimizer is None:
        optimizer = optim.AdamW(model.parameters(), lr=f12-3, betas=(0.9, 0.95), amsgrad=True)

    if val_loss_fn is None:
        val_loss_fn = loss_fn

#    if tensorboard_path is not None:
#        writer_train = SummaryWriter(tensorboard_path + "/train", flush_secs=1)

    # dataset_valid has priority over validation_fraction. if no dataset_valid
    # but validation fraction then build dataset_train and dataset_valid
    if dataset_valid is None and validation_fraction is not None:
        # split dataset into a training and validation set
        if validation_fraction <= 0 or validation_fraction >= 1:
            raise ValueError("validation_fraction should be in (0, 1).")

        n_samples = len(dataset_train)
        indices = np.arange(n_samples)
        if shuffle:
            numpy_rng.shuffle(indices)
        ind_split = int(np.floor((1 - validation_fraction) * n_samples))
        train_indices, val_indices = indices[:ind_split], indices[ind_split:]
        dataset_valid = torch_data.TensorDataset(*dataset_train[val_indices])
        dataset_train = torch_data.TensorDataset(*dataset_train[train_indices])

    if dataset_valid is not None:
        if is_packed_autoreg:
            dataloader = torch_data.DataLoader(
                dataset=dataset_valid, batch_size=dataset_valid.__len__()
            )
            X_valid, y_valid = next(iter(dataloader))
        else:
            X_valid = dataset_valid.tensors[0]
            y_valid = dataset_valid.tensors[1]

        if return_best_model:
            if is_packed_autoreg:
                best_per_dim_val_losses = None
            else:
                best_val_loss = np.inf

#        if tensorboard_path is not None:
#            writer_valid = SummaryWriter(tensorboard_path + "/valid", flush_secs=1)
        if log_loss:
            model.tracked_val_losses = []

    dataset_train = torch_data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        sampler=sampler,
    )

    n_train = len(dataset_train.dataset)

    val_scheduler = isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau) or (
        isinstance(scheduler, list)
        and isinstance(scheduler[0], optim.lr_scheduler.ReduceLROnPlateau)
    )

    if log_loss:
        model.tracked_losses = []

    if tensorboard_path is not None and hasattr(model, "alpha"):
        for i in range(model.alpha.shape[0]):
            writer_train.add_scalar(
                f"multi-step loss weights/alpha_{{i + 1}}", model.alpha[i].item(), -1
            )

    for epoch in tqdm(range(n_epochs), desc="Training epoch"):
        model.train()

        if scheduler is not None and not val_scheduler:
            if isinstance(scheduler, list):
                for i, s in enumerate(scheduler):
                    s.step()
            else:
                scheduler.step()

        train_loss = 0
        total_norm = 0.0
        if is_packed_autoreg:
            per_dim_losses = torch.zeros((model.n_preds))

        # training
        # for (i, (x, y)) in tqdm(enumerate(dataset_train), desc="Training batches", total=len(dataset_train)):
        for i, (x, y) in enumerate(dataset_train):
            x, y = x.to(device), y.to(device)
            x, y = Variable(x), Variable(y)
            model.zero_grad()

            if is_vae:
                out = model(y, x)
                loss = loss_fn(y, out)
                train_loss += len(x) * loss.item()
            elif is_packed_autoreg:
                out = model(y, x)
                loss, per_dim_loss = loss_fn(y, out)
                per_dim_losses += len(x) * per_dim_loss
                train_loss += len(x) * loss.item()
            elif is_nvp:
                loss = -loss_fn(y, x).sum()
                train_loss += loss.item()
            else:
                out = model(x)
                loss = loss_fn(y, out)
                train_loss += len(x) * loss.item()

            if torch.isnan(x).any():
                print(f"at batch {{i}} x has NaN")

            if torch.isnan(y).any():
                print(f"at batch {{i}} y has NaN")

            loss.backward()

            # compute gradient norm
            norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    norm += param_norm.item() ** 2
            norm = norm ** (1.0 / 2)
            # sanity check
            if torch.isnan(torch.Tensor([norm])).any():
                print(f"at batch {{i}} the gradients are: {{norm}}")
                print("skip this batch")
            else:
                if clip_gradients:
                    # clip gradient norm
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), clip_gradients_value
                    )

                if isinstance(optimizer, list):
                    for o in optimizer:
                        o.step()
                else:
                    optimizer.step()

                total_norm += norm

        if verbose:
            print("===== gradient norm check (average norm across batches) =====")
            print(f"total norm: {{total_norm / n_train}}")
            print("\n")

        train_loss /= n_train
        if is_packed_autoreg:
            per_dim_losses /= n_train

        if log_loss:
            model.tracked_losses.append(train_loss)

        if verbose:
            if dataset_valid is None:
                print(
                    "[{{}}/{{}}] Training loss: {{:.4f}}".format(
                        epoch, n_epochs - 1, train_loss
                    )
                )
            else:
                print(
                    "[{{}}/{{}}] Training loss: {{:.4f}}".format(
                        epoch, n_epochs - 1, train_loss
                    ),
                    end="\t",
                )

        if tensorboard_path:
            writer_train.add_scalar(
                f'loss/dim: {{target_dim if target_dim is not None else "all"}}',
                train_loss,
                epoch,
            )
            writer_train.add_scalar(
                f'gradient_norm/dim: {{target_dim if target_dim is not None else "all"}}',
                total_norm / n_train,
                epoch,
            )
            writer_train.add_scalar(
                f'scheduler_lr/dim: {{target_dim if target_dim is not None else "all"}}',
                optimizer.param_groups[0]["lr"],
                epoch,
            )
            if hasattr(model, "alpha"):
                for i in range(model.alpha.shape[0]):
                    writer_train.add_scalar(
                        f"multi-step loss weights/alpha_{{i+1}}",
                        model.alpha[i].item(),
                        epoch,
                    )
            if is_packed_autoreg:
                for i, value in enumerate(per_dim_losses):
                    writer_train.add_scalar(f"{{i}}_dim_loss", value, epoch)

        if is_packed_autoreg:
            per_dim_val_losses = torch.zeros((model.n_preds))

        # loss on validation set
        if dataset_valid is not None:
            if not (is_nvp or is_packed_autoreg):
                y_valid_pred = predict(
                    model,
                    X_valid,
                    batch_size=batch_size_predict,
                    disable_cuda=disable_cuda,
                    verbose=0,
                    is_vae=is_vae,
                )
                if is_vae:
                    y_valid_pred = [y_valid_pred, *model.encode(y_valid, X_valid)]

                val_loss = val_loss_fn(y_valid, y_valid_pred).item()
            elif is_packed_autoreg:
                y_valid_pred = predict(
                    model,
                    dataset_valid,
                    batch_size=batch_size_predict,
                    disable_cuda=disable_cuda,
                    verbose=0,
                    is_vae=is_vae,
                    is_packed_autoreg=is_packed_autoreg,
                )

                val_loss, per_dim_val_losses = val_loss_fn(y_valid, y_valid_pred)
                val_loss = val_loss.item()
            else:
                model.eval()
                val_loss = 0
                for batch_idx, data_t in enumerate(dataset_valid):
                    cond_data = data_t[0].float()
                    cond_data = cond_data.to(device)
                    data_t = data_t[1]
                    data_t = data_t.to(device)
                    with torch.no_grad():
                        val_loss += (
                            -val_loss_fn(data_t, cond_data).mean().item()
                        )  # sum up batch loss

                val_loss = val_loss / len(dataset_valid)

            if log_loss:
                model.tracked_val_losses.append(val_loss)

            if verbose:
                print("Validation loss: {{:.4f}}".format(val_loss))

            if tensorboard_path:
                writer_valid.add_scalar(
                    f'loss/dim: {{target_dim if target_dim is not None else "all"}}',
                    val_loss,
                    epoch,
                )
                if is_packed_autoreg:
                    for i, value in enumerate(per_dim_val_losses):
                        writer_valid.add_scalar(f"{{i}}_dim_loss", value, epoch)

            if val_scheduler:
                if isinstance(scheduler, list):
                    for i, s in enumerate(scheduler):
                        s.step(per_dim_val_losses[i])
                else:
                    scheduler.step(val_loss)

            if return_best_model:
                if is_packed_autoreg:
                    if best_per_dim_val_losses is None:
                        best_per_dim_val_losses = (
                            torch.ones_like(per_dim_losses).detach().numpy()
                        )
                        best_per_dim_val_losses *= np.inf
                        best_model = copy.deepcopy(model)
                    for i in range(len(per_dim_losses)):
                        if per_dim_losses[i] < best_per_dim_val_losses[i]:
                            best_model.nets[i] = copy.deepcopy(model.nets[i])
                            best_per_dim_val_losses[i] = per_dim_losses[i]
                    best_val_loss = best_per_dim_val_losses.mean()

                else:
                    if val_loss < best_val_loss:
                        if isinstance(model, torch.jit.RecursiveScriptModule):
                            model.save("my_model")
                            best_model = torch.jit.load("my_model")
                            best_val_loss = val_loss
                        else:
                            best_model = copy.deepcopy(model)  # XXX I don't like this
                            best_val_loss = val_loss
                        best_model.selected_epoch = epoch

            if epoch in keep_checkpoints:
                if not os.path.exists(Path(tensorboard_path) / "model_checkpoints"):
                    os.makedirs(Path(tensorboard_path) / "model_checkpoints")
                torch.save(
                    model.state_dict(),
                    Path(tensorboard_path)
                    / "model_checkpoints"
                    / f"model_epoch_{{epoch}}.pth",
                )

            if early_stopping is not None:
                early_stopping(val_loss)
                if early_stopping.early_stop:
                    break

            current_lr = scheduler.optimizer.param_groups[0]['lr']
            if current_lr <= min_lr:
                break


    if tensorboard_path:
        ks1, ENCE1, RMSE1 = regression_reliability_diagram(
            model=model,
            dataset=dataset_valid,
            writer=writer_valid,
            target_dim=target_dim,
        )

        writer_global = SummaryWriter(
            os.environ["OUTPUT_DIR"] + "/global_model_training", flush_secs=1
        )
        writer_global.add_scalar(
            f'Calibration/Kolmogorov Smirnov statistics (h=1) - dim: {{target_dim if target_dim is not None else "all"}}',
            float(ks1),
            int(os.environ["CURRENT_EPOCH"]),
        )
        writer_global.add_scalar(
            f"Calibration/ENCE (Expected Normalized Calibration Error) - dim: "
            f'{{target_dim if target_dim is not None else "all"}}',
            float(ENCE1),
            int(os.environ["CURRENT_EPOCH"]),
        )
        writer_global.add_scalar(
            f'Validation Loss/NLL - dim: {{target_dim if target_dim is not None else "all"}}',
            float(best_val_loss),
            int(os.environ["CURRENT_EPOCH"]),
        )
        writer_global.add_scalar(
            f'Validation Loss/RMSE - dim: {{target_dim if target_dim is not None else "all"}}',
            float(RMSE1),
            int(os.environ["CURRENT_EPOCH"]),
        )

    # return best model and best val loss if we want it
    if (dataset_valid is not None) and return_best_model:
        if n_epochs == 0:  # we return the passed model
            best_model = model
            y_valid_pred = predict(
                best_model,
                X_valid,
                batch_size=batch_size_predict,
                disable_cuda=disable_cuda,
                verbose=0,
                is_packed_autoreg=is_packed_autoreg,
                is_vae=is_vae,
            )

            if is_vae:
                y_valid_pred = [y_valid_pred, model.encode(y_valid, X_valid)]
            best_val_loss = val_loss_fn(y_valid, y_valid_pred).item()

        if return_best_model:
            model = best_model
            val_loss = best_val_loss

    return model, val_loss

def predict(
    model,
    dataset,
    batch_size=None,
    disable_cuda=False,
    verbose=0,
    is_vae=False,
    is_packed_autoreg=False,
):
    """Predict outputs of dataset using trained model"""

    if batch_size is None:
        batch_size = len(dataset)

    model.eval()
    device = _set_device(disable_cuda=disable_cuda)

    dataset = torch_data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    predictions = []
    with torch.no_grad():
        # for i, x in tqdm(enumerate(dataset), desc="Validation batches", total=len(dataset)):
        for i, x in enumerate(dataset):
            if is_packed_autoreg:
                (x, y) = x
                y = y.to(device)
            x = x.to(device)
            if is_vae:
                predictions.append(model.sample(x).cpu())
            elif is_packed_autoreg:
                predictions.append(model.forward(y, x).cpu())
            else:
                predictions.append(model.forward(x).cpu())

            if verbose and i % 100 == 0:
                print("[{{}}/{{}}]".format(i, len(dataset)))

    return torch.cat(predictions, dim=0)
# mbrl-tools pytroch_train END

# Add custom initialization to all models
def create_linear(input_size: int = 32, output_size: int = 32, bias=True):
    layer = nn.Linear(input_size, output_size, bias=bias)

    # nn.init.kaiming_uniform_(layer.weight, nonlinearity='leaky_relu')
    # if bias:
    #    nn.init.zeros_(layer.bias)

    return layer


class CustomLoss:
    def __call__(self, y_true, y_pred):
        mus = y_pred[: len(y_true)]
        sigmas = y_pred[len(y_true) : len(y_true) * 2]
        w = y_pred[2 * len(y_true) :]

        if str(nn_type) == "NN":
            mean_sigmas = torch.mean(sigmas)
            sigmas = mean_sigmas.repeat(sigmas.shape)

        # the torch distributions expects (batch_size, n_gaussians, observation_dim)
        mus = mus.reshape((len(y_true), int(n_gaussians), -1))
        sigmas = sigmas.reshape((len(y_true), int(n_gaussians), -1))

        # w is a vector of ones of shape (n_samples, 1) in this case

        mix = D.Categorical(w)
        comp = D.Independent(D.Normal(mus, sigmas), 1)
        gmm = D.MixtureSameFamily(mix, comp)
        # this distribution is equivalent to: D.Independent(D.Normal(np.swapaxes(mus,1,2), np.swapaxes(sigmas,1,2)), 2)

        # probs = gauss_pdf(y_true, mus, sigmas)
        # summed_prob = torch.sum(probs * w, dim=1)
        likelihood = gmm.log_prob(y_true)

        # clamp summed_prob to avoid zeros when taking the log
        # eps = torch.finfo(summed_prob.dtype).eps
        # summed_prob = torch.clamp(summed_prob, min=eps)

        # nll = -torch.log(summed_prob)
        # nll = torch.mean(nll)
        nll = -likelihood.mean()
        return nll


class Regressor(BaseGenerativeRegressor):
    def __init__(self, metadata):
        self.metadata = metadata
#        self.max_dists = max_dists
        self.decomposition = "autoregressive"
        self.target_dim = 1
        self.det_sample = str(sampling) == "det"

    def fit(self, X_in, y_in):
#        ind = np.arange(len(X_in))
#        selected_ind = np.unique(
#            np.hstack((ind[: int(first_n_steps)], ind[-int(last_n_steps) :]))
#        )
#        X_in = X_in[selected_ind]
#        y_in = y_in[selected_ind]
        y_in = y_in.astype(float)       
        if self.metadata["score_name"] == "rmsle":
            y_in = np.log1p(y_in)
        if bool(is_diff):
            y_in = y_in - X_in[:, self.target_dim].reshape((-1, 1))

        if float(y_scale) > 0.0:
            self.y_mean = y_in.mean()
            self.y_std = y_in.std()
            y_in -= self.y_mean
            y_in = float(y_scale) * y_in / self.y_std

        self.model = SimpleBinnedNoBounds(int(n_gaussians), X_in.shape[1])

        dataset = torch.utils.data.TensorDataset(torch.Tensor(X_in.to_numpy()), torch.Tensor(y_in))
        optimizer = optim.AdamW(
            self.model.parameters(), lr=float(learning_rate), betas=(0.9, 0.95), amsgrad=True
        )


        # change patience
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            "min",
            factor=0.1,
            patience=10,
            cooldown=5,
            min_lr=1e-8,
            verbose=True,
        )
        scheduler = SuperVerboseReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=5,
            threshold=0.0,
            threshold_mode='abs',
            cooldown=0,
            min_lr=1e-5,
            verbose=True,
        )

        loss = CustomLoss()

        # added tensorboard path
        # extra = 'default_6007'
        # tensorboard_path_specific = 'scheduler_patience'
        batch_size = max(16, min(int(2 ** (3 + np.floor(np.log10(len(X_in))))), 512))
        print(f"Batch size = {{batch_size}}")
        self.model, _ = train(
            self.model,
            dataset,
            validation_fraction=VALIDATION_FRACTION,
            n_epochs=int(n_epochs),
            batch_size=int(batch_size),
            loss_fn=loss,
            optimizer=optimizer,
            #early_stopping=EarlyStopping(patience=10, cooldown=5, min_delta=0),
            scheduler=scheduler,
            min_lr=1e-5, 
            return_best_model=True,
            disable_cuda=True,
            drop_last=True,
            verbose=True,
        )

    def predict(self, X):
        # we use predict sequentially in RL and there is no need to compute
        # model.eval() each time if the model is already in eval mode
        if self.model.training:
            self.model.eval()

        with torch.no_grad():
            X = torch.Tensor(X.to_numpy())
            n_samples = X.shape[0]
            y_pred = self.model(X)

            mus = y_pred[:n_samples].detach().numpy()
            sigmas = y_pred[n_samples : 2 * n_samples].detach().numpy()
            weights = y_pred[2 * n_samples :].detach().numpy()

        if str(nn_type) == "NN":
            mean_sigmas = sigmas.mean()
            sigmas = np.ones(sigmas.shape) * mean_sigmas

        if float(y_scale) > 0:
            mus *= self.y_std / float(y_scale)
            mus += self.y_mean
            sigmas *= self.y_std / float(y_scale)

        if bool(is_diff):
            mus += X[:, self.target_dim].detach().cpu().numpy().reshape((-1, 1))

        if self.det_sample:
            # we want to predict by the mean so we return only one Gaussian with
            # the mean of the mixture and a very small variance
            means = np.sum(mus * weights, axis=1).reshape((len(mus), 1))
            sigmas = np.full(shape=(n_samples, 1), fill_value=1e-10)

            params = np.concatenate((means, sigmas), axis=1)
            types = ["norm"]  # Gaussians
            weights = np.ones((n_samples, 1))
        else:
            # We put each mu next to its sigma
            params = np.empty((len(X), int(n_gaussians) * 2))
            params[:, 0::2] = mus
            params[:, 1::2] = sigmas
            types = ["norm"] * int(n_gaussians)

        y_pred = means
        if self.metadata["score_name"] == "rmsle":
            y_pred = np.expm1(y_pred)
        return y_pred
        
#        return weights, types, params

    def sample(self, X, rng=None, restart=None):
        n_samples = X.shape[0]
        rng = check_random_state(rng)

        distribution = self.predict(X)

        weights, _, params = distribution
        means = params[:, 0::2]
        sigmas = params[:, 1::2]

        weights /= np.sum(weights, axis=1)[:, np.newaxis]
        # vectorize sampling of one component for each sample
        cum_weights = weights.cumsum(axis=1)
        sampled_components = (cum_weights > rng.rand(n_samples)[:, np.newaxis]).argmax(
            axis=1
        )
        # get associated means and sigmas
        sampled_means = means[np.arange(n_samples), sampled_components]
        sampled_sigmas = sigmas[np.arange(n_samples), sampled_components]

        y_sampled = sampled_means + rng.randn(n_samples) * sampled_sigmas
        return y_sampled


class SimpleBinnedNoBounds(nn.Module):
    def __init__(self, n_sigmas, input_size):
        super(SimpleBinnedNoBounds, self).__init__()
        output_size_sigma = n_sigmas
        output_size_mus = n_sigmas

        # activation = nn.LeakyReLU
        activation = nn.Tanh

        self.linear0 = create_linear(input_size, int(layer_size))
        self.act0 = activation()
        self.drop = nn.Dropout(p=float(drop_first))

        self.common_block = nn.Sequential()
        for i in range(int(n_layers)):
            self.common_block.add_module(
                f"layer{{i + 1}}-lin", create_linear(int(layer_size), int(layer_size))
            )
            self.common_block.add_module(
                f"layer{{i + 1}}-bn", nn.BatchNorm1d(int(layer_size))
            )
            self.common_block.add_module(f"layer{{i + 1}}-act", activation())
            if i % 2 == 0:
                self.common_block.add_module(
                    f"layer{{i + 1}}-drop", nn.Dropout(p=float(drop_repeated))
                )

        self.mu = nn.Sequential(
            create_linear(int(layer_size), int(layer_size)),
            activation(),
            create_linear(int(layer_size), output_size_mus),
        )

        self.sigma = nn.Sequential(
            create_linear(int(layer_size), int(layer_size)),
            activation(),
            create_linear(int(layer_size), output_size_sigma),
            # nn.Sigmoid()
        )

        self.w = nn.Sequential(
            create_linear(int(layer_size), n_sigmas), nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.linear0(x)

        # debugguing nan values in loss
        # if torch.isnan(x).any():
        #     print('weights have nan:')
        #     print(f'{{torch.isnan(self.linear0.weight).any()}}')
        #     print(f'biases have nan: {{torch.isnan(self.linear0.bias).any()}}')
        #     raise ValueError(f"got nan at 'x = self.linear0(x)'")

        x = self.act0(x)
        raw = self.drop(x)
        x = self.common_block(raw)
        x = x + raw
        mu = self.mu(x)
        sigma = self.sigma(x)
        if str(sigma_transf) == "asym_sigmoid":
            # to make prior sigma ~ 1 (output should be scaled)
            sigma = 1 / (1 + torch.exp(-(sigma / 10 + 2)))
        elif str(sigma_transf) == "exp":
            sigma = torch.exp(sigma)
        else:
            raise ValueError(f"{{str(sigma_transf)}}: uknown sigma transformation")
        w = self.w(x)
        return torch.cat([mu, sigma, w], dim=0)
