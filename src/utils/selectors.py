from typing import Dict, Callable, Union

import pymc as pm
from pymc import Model

from src.generators.mixture_generator import (
    UnivariateGaussianMixtureGenerator
)


def step_selector(
        sampler: str,
        sampler_params: Dict,
        model: pm.Model
) -> Union[Callable, Model]:
    """
    Selects and returns the appropriate step function for the given sampler type.

    Args:
        sampler (str): The type of sampler.
        sampler_params (Dict): The parameters for the sampler.
        model (pm.Model): The PyMC model.

    Returns:
        Union[Callable, Model]: The selected step function.

    Raises:
        ValueError: If an invalid sampler type is provided.
    """
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
    """
    Selects and returns the appropriate generator function for the given generator type.

    Args:
        type_of_generator (str): The type of generator.

    Returns:
        Callable: The selected generator function.

    Raises:
        ValueError: If an invalid generator type is provided.
    """
    if type_of_generator == 'univariate':
        generator = UnivariateGaussianMixtureGenerator
    else:
        raise ValueError("Invalid generator type")
    return generator
