import argparse
from datetime import datetime
import json
import warnings
import os

import arviz as az
from dacite import from_dict
import pymc as pm
from pymc.util import get_value_vars_from_user_vars
import matplotlib.pyplot as plt


from src.utils.selectors import (
    step_selector,
    generator_selector
)

from src.utils.gmm_parameters import (
    UnivariateGMMParameters,
    UnivariateGMMPriorParameters
)

from src.models.model_builder import build_model

from src.utils.gibbs_sampler import (
    UnivariateGibbsSampler
)

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--type', choices=['univariate'],
                    default='univariate', help='Type of simulation: univariate or multivariate')
parser.add_argument(
    '--sampler',
    choices=['Metropolis', 'HMC', 'Gibbs', 'NUTS'],
    default='Metropolis',
    help='Type of sampler to use')

dataclass_dict = {
    "univariate": UnivariateGMMParameters
}

prior_dataclass_dict = {
    "univariate": UnivariateGMMPriorParameters
}

if __name__ == "__main__":
    args = parser.parse_args()
    experiment_type = args.type
    sampler = args.sampler

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    with open('configs/experiment_params.json') as f:
        experiment_params = json.load(f)

    with open('configs/sampler_params.json') as f:
        sampler_params = json.load(f)

    with open('configs/prior_params.json') as f:
        prior_params_dict = json.load(f)

    prior_params = from_dict(data_class=prior_dataclass_dict[experiment_type],
                             data=prior_params_dict[experiment_type])

    traceplot_dir = f"results/{timestamp}/figures"
    summary_stats_dir = f"results/{timestamp}/data"
    os.makedirs(traceplot_dir, exist_ok=True)
    os.makedirs(summary_stats_dir, exist_ok=True)

    for case_name, params in experiment_params[experiment_type].items():
        print(f"case: {case_name}, params: {params}")

        # initialize the data generator
        simulation_gmm_params = from_dict(data_class=dataclass_dict[experiment_type],
                                          data=params)
        generator_distribtuion = generator_selector(experiment_type)
        data_generator = generator_distribtuion(simulation_gmm_params)
        sample = data_generator.generate()

        # build the model
        model = build_model(
            experiment_type,
            sample,
            prior_params,
            simulation_gmm_params
        )

        # sample from the model
        with model:
            if sampler == 'Gibbs' and experiment_type == 'univariate':
                value_vars = get_value_vars_from_user_vars([model.mu0, model.mu_k_raw, model.sigma_k], model)
                step_method = UnivariateGibbsSampler(
                    vars=value_vars,
                    model=model,
                    y=sample,
                    pi=simulation_gmm_params.weights,
                    theta=prior_params.mu0_mean,
                    nu=prior_params.mu0_std,
                    tau2=prior_params.muk_variance,
                    alpha0=prior_params.sigma0_alpha,
                    beta0=prior_params.sigma0_beta
                )
            else:
                step_method = step_selector(sampler, sampler_params[sampler], model)
            trace = pm.sample(10000, tune=1000, chains=4, step=step_method)

        # plot the results
        axes = az.plot_trace(trace)
        fig = axes[0, 0].figure
        fig.suptitle(f"Traceplot for {case_name}", fontsize=16)
        traceplot_filename = os.path.join(f"{traceplot_dir}", f"{sampler}-{case_name}_traceplot.png")
        plt.savefig(traceplot_filename, dpi=300)
        plt.close(fig)
        print(f"Traceplot saved to {traceplot_filename}")

        # save summary as json
        summary_df = pm.summary(trace)
        summary_dict = summary_df.to_dict(orient="index")
        summary_filename = os.path.join(f"{summary_stats_dir}", f"{case_name}_summary.json")
        with open(summary_filename, "w") as f:
            json.dump(summary_dict, f, indent=4)
        print(f"Summary saved to {summary_filename}")
