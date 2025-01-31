from typing import List, Dict, Union, NoReturn

import numpy as np
from pymc import Model
from pymc.step_methods.arraystep import BlockedStep
from scipy.stats import norm, multivariate_normal, invwishart

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
        K = self.n_components

        for component in range(K):
            idx = np.where(z == component)
            y_k = self.y[idx]
            N_k = len(y_k)
            var_post = 1/(N_k/sigma2[component] + 1/tau2)
            mean_post = var_post * (y_k.sum()/sigma2[component] + mu0/tau2)
            mu[component] = np.random.normal(mean_post, np.sqrt(var_post))
        return mu

    def _step_sigma_k(self,
                      mu: np.ndarray,
                      sigma2: np.ndarray,
                      z: np.ndarray) -> np.ndarray:
        K = self.n_components
        
        for component in range(K):
            idx = np.where(z == component)
            y_k = self.y[idx]
            N_k = len(y_k)
            alpha_post = self.alpha0 + N_k/2
            beta_post = self.beta0 + 0.5 * np.sum((y_k - mu[component])**2)
            sigma2[component] = np.random.gamma(alpha_post, 1/beta_post)
        
        return sigma2

    def _step_mu_0(self,
                   mu0: np.ndarray,
                   mu: np.ndarray) -> np.ndarray:
        K = self.n_components
        var_post = 1/(K/self.tau2 + 1/self.nu)
        mean_post = var_post * (self.theta/self.nu + mu.sum()/self.tau2)
        mu0 = np.random.normal(mean_post, np.sqrt(var_post))
        return mu0

    def _step_z(self,
                mu: np.ndarray,
                sigma2: np.ndarray) -> np.ndarray:
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

class MultivariateGibbsSampler(BlockedStep):
    def __init__(
            self, 
            vars: List[str], 
            model: Model, 
            y: np.ndarray,
            pi: np.ndarray,
            prior_vars: np.ndarray,
            hyperprior_mean: np.ndarray,
            hyperprior_vars: np.ndarray) -> NoReturn:
        super().__init__()
        self.y = y
        self.pi = pi
        self.gamma = prior_vars
        self.hyperprior_mean = hyperprior_mean
        self.hyperprior_vars = hyperprior_vars
        self.n_components = len(self.pi)
        self.n_samples = len(self.y)
        self.dimension = self.y.shape[1]
        self.vars = vars

    def step(self,
             point: Dict[str, Union[float, np.ndarray]]) -> Union[Dict[str, np.ndarray], List]:
        #print(point)
        mu0_current = point['mu0']
        mu_current = point['mu_k_raw']
        #print("packed chol (inside step)", point['packed_chol'])
        sigma2_current = self._expand_packed_chol(point['packed_chol'])
        #print("old packed chol", point['packed_chol'])
        #print(sigma2_current)
        z_new = self._step_z(mu_current, sigma2_current)
        mu_new = self._step_mu_k(mu0_current, mu_current, sigma2_current, z_new)
        mu0_new = self._step_mu_0(mu_current)
        sigma2_new = self._step_sigma_k(sigma2_current, z_new)
        packed_col_new = self.__pack_variance_matrix(sigma2_new)
        #print("new", sigma2_new)
        new_point = point.copy()
        new_point['mu0'] = mu0_new
        new_point['mu_k_raw'] = mu_new
        #print("fine updating means")
        #print("new packed cholesky", packed_col_new)
        new_point['packed_chol'] = packed_col_new
        return new_point, []

    def _step_z(self,
                mu: np.ndarray,
                sigma2: np.ndarray) -> np.ndarray:
        N = self.n_samples
        K = self.n_components
        P = self.dimension
        #print("Y: \n", self.y)
        log_probs = np.zeros((N, K))
        #print("at beginnning: \n", probs)
        eigs = np.linalg.eigvals(sigma2)
        print("sigma2 eignevalues", eigs)
        for k in range(K):
            #print("inside loop", mu[k], "sigma", sigma2[k])
            # il problema è sigma
            # lo devo mettere in una lista di array
            #print(multivariate_normal.pdf(self.y, mean=mu[k], cov=sigma2[k]))
            #print("covarianca", sigma2)
            log_probs[:, k] = np.log(self.pi[k]) + multivariate_normal.logpdf(self.y, mean=mu[k], cov=sigma2) # this is giving always zero probabilities
            print("probs[:k]", log_probs[:, k])
        
        #print("probs (before sum)", probs)
        #probs_sum = probs.sum(axis=1, keepdims=True)
        #probs_sum[probs_sum == 0] = 1e-16
        #print("probs_sum", probs_sum)
        #probs /= probs_sum
        probs = np.exp(log_probs - log_probs.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        # error is caused when all of the probs are 0
        #print("probs", probs)
        #print("probs", probs.shape)
        # error now comes from the fact that the probabilities are all zero, so they dont sum to 1
        z_new = np.array([np.random.multinomial(1, probs[i, :]).argmax() for i in range(N)])

        return z_new
    
    def _step_mu_k(self,
                   mu0: np.ndarray,
                   mu: np.ndarray,
                   sigma_k: np.ndarray,
                   z: np.ndarray) -> np.ndarray:
        K = self.n_components

        for component in range(K):
            idx = np.where(z == component)
            y_k = self.y[idx]
            N_k = len(y_k)
            sigma_k_inv = np.linalg.inv(sigma_k)
            gamma_inv = np.linalg.inv(self.gamma)
            var_post = np.linalg.inv(N_k * sigma_k_inv- gamma_inv)
            mean_post = var_post @ (N_k * sigma_k_inv @ y_k.sum(axis=0) + gamma_inv @ mu0)
            mu[component] = np.random.multivariate_normal(mean_post, var_post)
        return mu
    
    def _step_sigma_k(self,
                      sigma: np.ndarray,
                      z: np.ndarray) -> np.ndarray:
        K = self.n_components

        #for component in range(K):
        #    idx = np.where(z == component)
        #    y_k = self.y[idx]
        #    N_k = len(y_k)
        #    S_k = np.cov(y_k, rowvar=False)
        #    S_post = invwishart.rvs(df = N_k + self.dimension + 1, scale = S_k)
        #    sigma[component] = S_post
        S_k = np.cov(self.y, rowvar=False)
        S_post = invwishart.rvs(df=self.n_samples + self.dimension + 1 + 5, scale=S_k + 1e-6 * np.eye(self.dimension)) # 5 is to regularize
        return S_post
    
    def _step_mu_0(self, mu_current):
        gamma_inv = np.linalg.inv(self.gamma)
        hyp_var_inv = np.linalg.inv(self.hyperprior_vars)
        var_post = np.linalg.inv(gamma_inv + hyp_var_inv)
        mean_post = var_post @ (gamma_inv @ mu_current.sum(axis = 0) + hyp_var_inv @ self.hyperprior_mean)
        mu0 = np.random.multivariate_normal(mean_post, var_post)
        return mu0
    
    def _expand_packed_chol(self, packed_chol: np.ndarray) -> np.ndarray:
        #print("--- INSIDE _expand_packed_chol ---")
        #print("packed_chol:", packed_chol)

        D = self.dimension
        n_elem_per_cov = D * (D + 1) // 2

        # Validate input size
        if packed_chol.shape[0] != n_elem_per_cov:
            raise ValueError(
                f"Expected packed_chol to have {n_elem_per_cov} elements, but got {packed_chol.shape[0]}."
            )

        # Initialize lower triangular matrix
        chol_matrix = np.zeros((D, D))
        
        idx = np.tril_indices(D)  # Indices for lower triangular part
        chol_matrix[idx] = packed_chol  # Fill only lower triangular part

        #print("Expanded chol_matrix:", chol_matrix)
        
        return chol_matrix
    
    def __pack_variance_matrix(self, variance_matrix: np.ndarray) -> np.ndarray:
        if variance_matrix.shape[0] != variance_matrix.shape[1]:
            raise ValueError("variance_matrix must be a square (D, D) matrix.")

        D = variance_matrix.shape[0]

        # Compute Cholesky decomposition
        L = np.linalg.cholesky(variance_matrix)  # Lower triangular (D, D)
        
        # Extract lower-triangular elements
        idx = np.tril_indices(D)
        packed_chol = L[idx]  # (D * (D + 1) // 2,)

        return packed_chol