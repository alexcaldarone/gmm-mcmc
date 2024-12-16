# Comparing MCMC Algorithms for Bayesian Hierarchical Gaussian Mixture Models

## Folder structure
```
gmm-mcmc/
  ├── src/
  │   ├── generators/ # code to generate gaussian mixtures
  │   ├── models/ # code to build and validate hierarchical models
  │   ├── utils/
  ├── configs/ # configuration files for experiments
  ├── results/ # experiment resutls
  ├── requirements.txt
  ├── main.py # file to run the experiments
  ├── report.pdf
  └── README.md
```

## Instructions
### 1. Creating the environment
First of all, clone the repository and create a virtual environment by running the following commands,
```bash
git clone https://github.com/alexcaldarone/gmm-mcmc.git
cd gmm-mcmc 
python3 -m venv venv
```

To activate the environment, if on Windows use:
```bash
venv\Scripts\activate
```

If on MacOs/Linnix use:
```bash
source venv/bin/activate
```

Then install the dependencies by running,
```bash
pip install -r requirements.txt
```

### 2. Running the experiments
The project allows one to easily run the experiments needed. Here is a breakdown of the steps needed:

1. Change the experiment settings (parameters of the components of the mixture) in the `experiment_params.json` file (__Note:__ Change only the numerical values, the structure of the dictionary should stay the same)

2. Change the prior parameters of the hierarchical model components in the `prior_params.json` file (the same note as above holds - only change the numerical values)

3. Run the experiments using the command line by calling:

The `main.py` allws the user to run the experiments by specifying what set of experiments we are interested in and the sampler to use. The options are:
- Experiment tyepe: `univariate` (runs the experiments for the mixture of univariate gaussians) or `multivariate` (runs the experiments for the mixture of multivariate gaussian random variables)
- Sampler: which sampler to use for the experiments. At the moment the supported options are: `Metropolis` (to use Metropolis-Hastings algorithm) and `HMC` (to use Hamiltonian Monte Carlo)

In practice, the experiments are run by calling:

```bash
python3 main.py --type [type] --sampler [sampler]
```
Running this command will start the sampling for all the settings specified in the `experiment_params.json` file. The results will be saved in the `results` directory, under a folder of the format `YYYYMMDD_HHMMSS`. In here you will be able to find the traceplots and the summary statistics returned by the sampler.

If you wish to run more than one sampler for the same experiment settings, you will have to run one at a time (so run the above command twice, by changing the sampler option).

If you need extra help on how to manage the command line options, run
```bash
python3 main.py --help
```
