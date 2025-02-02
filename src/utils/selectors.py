from typing import Dict, Callable, Union

import pymc as pm
from pymc import Model

from src.generators.mixture_generator import (
    UnivariateGaussianMixtureGenerator
)

from .gibbs_sampler import (
    UnivariateGibbsSampler
)

def step_selector(
        sampler: str,
        sampler_params: Dict,
        model: pm.Model
    ) -> Union[Callable, Model]:
    print("sampler selected:", sampler)
    if sampler == 'Metropolis':
        # default proposal distribution is Normal
        step = pm.Metropolis
    elif sampler == 'HMC':
        step = pm.HamiltonianMC
    elif sampler == "NUTS":
        step = pm.NUTS
    else:
        raise ValueError("Invalid sampler type")
    
    if not sampler_params:
        return step()
    
    return step(**{k: v for k, v in sampler_params.items() if v is not None})

def generator_selector(
        type_of_generator: str
    ) -> Callable:
    if type_of_generator == 'univariate':
        generator = UnivariateGaussianMixtureGenerator
    else:
        raise ValueError("Invalid generator type")
    return generator