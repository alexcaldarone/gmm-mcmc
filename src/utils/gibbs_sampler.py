from typing import List, Dict, Union, NoReturn

import numpy as np
from pymc import Model
from pymc.step_methods.arraystep import BlockedStep
from scipy.stats import norm


class UnivariateGibbsSampler(BlockedStep):
    def __init__(
            self,
            vars: List[str],
            model: Model,
            y: np.ndarray,
            pi: np.ndarray,
            theta: float,
            nu: float,
            tau2: float,
            alpha0: float,
            beta0: float) -> NoReturn:
        """
        Initialize the UnivariateGibbsSampler.

        Args:
            vars (List[str]): List of variable names.
            model (Model, optional): The model. Defaults to None.
            y (np.ndarray, optional): The observed data. Defaults to None.
            pi (np.ndarray, optional): The mixture weights. Defaults to None.
            theta (float, optional): Hyperparameter for mu0. Defaults to None.
            nu (float, optional): Hyperparameter for mu0. Defaults to None.
            tau2 (float, optional): Hyperparameter for mu0. Defaults to None.
            alpha0 (float, optional): Hyperparameter for sigma2. Defaults to None.
            beta0 (float, optional): Hyperparameter for sigma2. Defaults to None.
        """
        super().__init__()
        self.y = y
        self.pi = pi
        self.theta = theta
        self.nu = nu
        self.tau2 = tau2
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.n_components = len(self.pi)
        self.n_samples = len(self.y)
        self.vars = vars

    def step(self,
             point: Dict[str, Union[float, np.ndarray]]) -> Union[Dict[str, np.ndarray], List]:
        """
        Perform a Gibbs sampling step.

        Args:
            point (Dict[str, Union[float, np.ndarray]]): Current parameter values.

        Returns:
            Union[Dict[str, np.ndarray], List]: Updated parameter values and empty list.
        """
        mu0_current = point['mu0']
        mu_current = point['mu_k_raw']
        sigma2_current = point['sigma_k']

        z_new = self._step_z(mu_current, sigma2_current)
        mu_new = self._step_mu_k(mu0_current, mu_current, sigma2_current, z_new, self.tau2)
        mu0_new = self._step_mu_0(mu0_current, mu_new)
        sigma2_new = self._step_sigma_k(mu_new, sigma2_current, z_new)

        new_point = point.copy()
        new_point['mu0'] = mu0_new
        new_point['mu_k_raw'] = mu_new
        new_point['sigma_k'] = sigma2_new
        return new_point, []

    def _step_mu_k(self,
                   mu0: np.ndarray,
                   mu: np.ndarray,
                   sigma2: np.ndarray,
                   z: np.ndarray,
                   tau2: np.ndarray) -> np.ndarray:
        """
        Perform a Gibbs sampling step for updating mu_k.

        Args:
            mu0 (np.ndarray): Current value of mu0.
            mu (np.ndarray): Current value of mu_k.
            sigma2 (np.ndarray): Current value of sigma_k.
            z (np.ndarray): Current value of z.
            tau2 (np.ndarray): Current value of tau2.

        Returns:
            np.ndarray: Updated value of mu_k.
        """
        K = self.n_components

        for component in range(K):
            idx = np.where(z == component)
            y_k = self.y[idx]
            N_k = len(y_k)
            var_post = 1 / (N_k / sigma2[component] + 1 / tau2)
            mean_post = var_post * (y_k.sum() / sigma2[component] + mu0 / tau2)
            mu[component] = np.random.normal(mean_post, np.sqrt(var_post))
        return mu

    def _step_sigma_k(self,
                      mu: np.ndarray,
                      sigma2: np.ndarray,
                      z: np.ndarray) -> np.ndarray:
        """
        Perform a Gibbs sampling step for updating sigma_k.

        Args:
            mu (np.ndarray): Current value of mu_k.
            sigma2 (np.ndarray): Current value of sigma_k.
            z (np.ndarray): Current value of z.

        Returns:
            np.ndarray: Updated value of sigma_k.
        """
        K = self.n_components

        for component in range(K):
            idx = np.where(z == component)
            y_k = self.y[idx]
            N_k = len(y_k)
            alpha_post = self.alpha0 + N_k / 2
            beta_post = self.beta0 + 0.5 * np.sum((y_k - mu[component])**2)
            sigma2[component] = np.random.gamma(alpha_post, 1 / beta_post)

        return sigma2

    def _step_mu_0(self,
                   mu0: np.ndarray,
                   mu: np.ndarray) -> np.ndarray:
        """
        Perform a Gibbs sampling step for updating mu0.

        Args:
            mu0 (np.ndarray): Current value of mu0.
            mu (np.ndarray): Current value of mu_k.

        Returns:
            np.ndarray: Updated value of mu0.
        """
        K = self.n_components
        var_post = 1 / (K / self.tau2 + 1 / self.nu)
        mean_post = var_post * (self.theta / self.nu + mu.sum() / self.tau2)
        mu0 = np.random.normal(mean_post, np.sqrt(var_post))
        return mu0

    def _step_z(self,
                mu: np.ndarray,
                sigma2: np.ndarray) -> np.ndarray:
        """
        Perform a Gibbs sampling step for updating z.

        Args:
            mu (np.ndarray): Current value of mu_k.
            sigma2 (np.ndarray): Current value of sigma_k.

        Returns:
            np.ndarray: Updated value of z.
        """
        N = self.n_samples
        K = self.n_components

        probs = np.zeros((N, K))

        for k in range(K):
            probs[:, k] = self.pi[k] * norm.pdf(self.y, loc=mu[k], scale=np.sqrt(sigma2[k]))

        # Normalize the probabilities row-wise
        probs_sum = probs.sum(axis=1, keepdims=True)
        probs_sum[probs_sum == 0] = 1e-16  # Avoid division by zero
        probs /= probs_sum

        z_new = np.array([np.random.choice(K, p=probs[i, :]) for i in range(N)])

        return z_new
