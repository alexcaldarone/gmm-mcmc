import numpy as np
from typing import NoReturn

from ..utils.gmm_parameters import (
    UnivariateGMMParameters
)


class UnivariateGaussianMixtureGenerator:
    """
    Generates samples from a univariate Gaussian mixture model.
    """

    def __init__(self, params: UnivariateGMMParameters) -> NoReturn:
        """
        Initializes the UnivariateGaussianMixtureGenerator.

        Args:
            params (UnivariateGMMParameters): The parameters of the Gaussian mixture model.
        """
        self.params = params

    def generate(self) -> np.ndarray:
        """
        Generates samples from the Gaussian mixture model.

        Returns:
            np.ndarray: The generated samples.
        """
        component_indices = np.random.choice(
            self.params.n_components,
            size=self.params.n_samples,
            p=self.params.weights
        )

        means = self.params.means[component_indices]
        stds = self.params.standard_deviations[component_indices]

        data = np.random.normal(loc=means, scale=stds)
        return data
