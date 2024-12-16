from typing import Dict, Callable, Union

import pymc as pm
from pymc import Model

from src.generators.mixture_generator import (
    UnivariateGaussianMixtureGenerator,
    MultivariateGaussianMixtureGenerator
)

from .gibbs_sampler import (
    UnivariateGibbsSampler
)

def step_selector(
        sampler: str,
        sampler_params: Dict
    ) -> Union[Callable, Model]:
    if sampler == 'Metropolis':
        # default proposal distribution is Normal
        step = pm.Metropolis
    elif sampler == 'HMC':
        step = pm.HamiltonianMC
    elif sampler == "Gibbs":
        step = UnivariateGibbsSampler
    else:
        raise ValueError("Invalid sampler type")
    return step

def generator_selector(
        type_of_generator: str
    ) -> Callable:
    if type_of_generator == 'univariate':
        generator = UnivariateGaussianMixtureGenerator
    elif type_of_generator == 'multivariate':
        generator = MultivariateGaussianMixtureGenerator
    else:
        raise ValueError("Invalid generator type")
    return generator