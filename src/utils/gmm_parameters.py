from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class UnivariateGMMParameters:
    """
    Class representing the parameters of a univariate Gaussian Mixture Model (GMM).

    Attributes:
        n_samples (int): The number of samples in the dataset.
        n_components (int): The number of components in the GMM.
        means (List[float]): The means of the Gaussian components.
        standard_deviations (List[float]): The standard deviations of the Gaussian components.
        weights (List[float]): The weights of the Gaussian components.

    Methods:
        __post_init__(): Validates and initializes the attributes of the class.
    """
    n_samples: int
    n_components: int
    means: List[float]
    standard_deviations: List[float]
    weights: List[float]

    def __post_init__(self):
        """
        Validates and initializes the attributes of the class.

        Raises:
            TypeError: If means, weights, or standard_deviations are not of type list.
            ValueError: If the number of means, weights, or standard_deviations does not match the number of components.
                        If the sum of weights is not equal to 1.
                        If any of the standard deviations are not positive.
        """
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