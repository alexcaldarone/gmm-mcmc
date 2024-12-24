from typing import Union
import numpy as np
import pymc as pm
import pytensor.tensor as pt

from src.utils.gmm_parameters import (
    UnivariateGMMParameters,
    UnivariateGMMPriorParameters,
    MultivariateGMMParameters,
    MultivariateGMMPriorParameters
)

def build_model(model_type: str, 
                sample: np.ndarray,
                prior_params: Union[UnivariateGMMPriorParameters, MultivariateGMMPriorParameters],
                gmm_params: Union[UnivariateGMMParameters, MultivariateGMMParameters]) -> pm.Model:
    if model_type not in ["univariate", "multivariate"]:
        raise ValueError("Invalid model type. Choose 'univariate' or 'multivariate'.")

    if model_type == "univariate":
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
    
    elif model_type == "multivariate":
        # note prior_params.dimension = gmm_params.dimension
        with pm.Model() as model:
            mu0 = pm.MvNormal("mu0", mu=prior_params.mu0_mean, cov=prior_params.mu0_std, shape=(prior_params.dimension,)) # have to add shape?

            # Priors for component means and variances
            mu_k_raw = pm.MvNormal("mu_k", mu = mu0, cov = prior_params.muk_variance, shape = (gmm_params.n_components, gmm_params.dimension))
            mu_k = pm.Deterministic("mu_k", mu_k_raw)
                        
            packed_chol = pm.LKJCholeskyCov("packed_chol", n=prior_params.dimension, eta=1, sd_dist=pm.HalfNormal.dist(2), 
                                            compute_corr=False)
            sigma_k = pm.expand_packed_triangular(prior_params.dimension, packed_chol, lower = False)
            
            # Define component distributions
            components = pm.MvNormal.dist(mu = mu_k, cov = sigma_k, shape = (gmm_params.n_components, gmm_params.dimension))

            # Likelihood
            y = pm.Mixture("y", w = gmm_params.weights, comp_dists = components, observed = sample)
    
    return model