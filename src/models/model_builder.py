from typing import Union
import numpy as np
import pymc as pm
import pytensor.tensor as pt

from src.utils.gmm_parameters import (
    UnivariateGMMParameters,
    UnivariateGMMPriorParameters
)

def build_model(model_type: str, 
                sample: np.ndarray,
                prior_params: UnivariateGMMPriorParameters,
                gmm_params: UnivariateGMMParameters) -> pm.Model:
    """
    Build a probabilistic model for Gaussian Mixture Model (GMM) based on the given parameters.

    Parameters:
    model_type (str): The type of the model. Choose between 'univariate' and 'multivariate'.
    sample (np.ndarray): The observed data sample.
    prior_params (UnivariateGMMPriorParameters): The prior parameters for the GMM.
    gmm_params (UnivariateGMMParameters): The parameters for the GMM.

    Returns:
    pm.Model: The built probabilistic model.

    Raises:
    ValueError: If an invalid model type is provided.
    """
    if model_type not in ["univariate"]:
        raise ValueError("Invalid model type.")

    with pm.Model() as model:
        # Hyper-priors
        mu0 = pm.Normal("mu0", mu=prior_params.mu0_mean, sigma=prior_params.mu0_std)
        
        # Priors for component means and variance
        mu_k_raw = pm.Normal("mu_k_raw", mu=mu0, sigma=prior_params.muk_variance, shape=gmm_params.n_components)
        mu_k = pm.Deterministic("mu_k", pt.sort(mu_k_raw))

        sigma_k = pm.InverseGamma("sigma_k", alpha=prior_params.sigma0_alpha, beta=prior_params.sigma0_beta, shape=gmm_params.n_components, transform=None)
        
        # Define component distributions
        components = [
            pm.Normal.dist(mu=mu_k[i], sigma=sigma_k[i], shape = (1,)) for i in range(gmm_params.n_components)
        ]
        
        # Likelihood
        y = pm.Mixture("y", w=gmm_params.weights, comp_dists=components, observed=sample)
    
    return model