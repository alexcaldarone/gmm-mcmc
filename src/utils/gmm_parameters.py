from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class UnivariateGMMParameters:
    n_samples: int
    n_components: int
    means: List[float]
    standard_deviations: List[float]
    weights: List[float]

    def __post_init__(self):
        if not isinstance(self.means, list):
            raise TypeError("Means must be a list")
        if len(self.means) != self.n_components:
            raise ValueError("The number of means must match the number of components")
        self.means = np.array(self.means)
        
        if not isinstance(self.weights, list):
            raise TypeError("Weights must be a list")
        if len(self.weights) != self.n_components:
            raise ValueError("The number of weights must match the number of components")
        if sum(self.weights) != 1:
            raise ValueError("The sum of weights must be equal to 1")
        self.weights = np.array(self.weights)
        
        if not isinstance(self.standard_deviations, list):
            raise TypeError("Standard deviations must be a list")
        if len(self.standard_deviations) != self.n_components:
            raise ValueError("The number of standard deviations must match the number of components")
        self.standard_deviations = np.array(self.standard_deviations)
        if np.any(self.standard_deviations <= 0):
            raise ValueError("Standard deviations must be positive")

@dataclass
class UnivariateGMMPriorParameters:
    mu0_mean: float
    mu0_std: float
    muk_variance: float # known variance for the component means
    sigma0_alpha: float # known alpha for the component variances
    sigma0_beta: float # known beta for the component variances