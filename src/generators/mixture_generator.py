import numpy as np
from typing import NoReturn

from ..utils.gmm_parameters import (
    UnivariateGMMParameters,
    MultivariateGMMParameters
)

class UnivariateGaussianMixtureGenerator:
    def __init__(self, params: UnivariateGMMParameters) -> NoReturn:
        self.params = params
    
    def generate(self) -> np.ndarray:
        component_indices = np.random.choice(
                    self.params.n_components,
                    size=self.params.n_samples,
                    p=self.params.weights
                )
        
        means = self.params.means[component_indices]
        stds = self.params.standard_deviations[component_indices]

        data = np.random.normal(loc=means, scale=stds)
        return data

class MultivariateGaussianMixtureGenerator:
    def __init__(self, params: MultivariateGMMParameters) -> NoReturn:
        self.params = params
    
    def generate(self) -> np.ndarray:
        component_indices = np.random.choice(
                    self.params.n_components,
                    size=self.params.n_samples,
                    p=self.params.weights
                )
        
        means = self.params.means[component_indices]

        data = np.array([np.random.multivariate_normal(mean=means[i], cov=self.params.covariance_matrix) for i in range(self.params.n_samples)])
        return data