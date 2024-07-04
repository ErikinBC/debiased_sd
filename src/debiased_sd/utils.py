"""
Utility helpers
"""

# Modules
import numpy as np
from typing import Callable, Tuple
from scipy.stats import kurtosis
from scipy.special import gamma, gammaln

##########################
# --- COMMON HELPERS --- #

def check_x_axis_ddof(
        x: np.ndarray, 
        axis: int | None = None,
        ddof:int = 1, 
        ) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Checks that x aligns with with the stated axis, and that ddof is valid

    Args
    ====
    x: np.ndarray
        An array or array-like object of arbitrary dimension: x.shape = (d1, d2, ..., dk)
    axis: int | None = None
        Axis to calculate the SD over
    ddof: int = 1
        The degrees of freedom for the sample SD

    Returns
    =======
    (x, std, n) or (np.array(x), sd_array[-axis] , x.shape[-axis])
    """
    # Input checks
    assert axis >= 0, f'Axis must be at least 0, not {axis}'
    assert ddof >= 0, f'Degrees of freedom must be >= 0, not {ddof}'
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    shape_len = len(x.shape)
    assert shape_len >= 1, f'You must provide at least a vector'
    if shape_len >= 2:
        valid_axes = list(range(shape_len))
        assert axis in valid_axes, f'axis must be one of {valid_axes}, not {axis}'
    # Calculate sample size
    sample_size = x.shape[axis]
    assert sample_size > ddof, f'If you have {sample_size} observations, then ddof must be at most {sample_size-1}, not {ddof}'
    # Calculate the sample SD
    sighat = np.std(x, axis=axis, ddof=ddof)
    return x, sighat, sample_size


###############################
# --- GAUSSIAN ADJUSTMENT --- #

def sd_gaussian(
        x: np.ndarray, 
        axis: int | None = None,
        ddof: int = 1,
        approx:bool = True,
        **kwargs,
        ) -> np.ndarray:
    """
    Calculate the de-biased sample SD when the data is drawn from a normal distribution
    
    Args
    ====
    x : np.ndarray
        An array or array-like object of arbitrary dimension: x.shape = (d1, d2, ..., dk)
    axis : int, optional
        Axis to calculate the SD over
    ddof : int, optional
        The degrees of freedom for the sample SD
    approx : bool, optional
        Should a log-approximation be used for the Gamma function calculation? Recommended if n is large
    """
    # Input checks
    x, sighat, n = check_x_axis_ddof(x, axis, ddof)
    # Calculate adjustment ratio
    if approx:
        gamma_ratio = np.exp(gammaln((n-1)/2) - gammaln(n/2))
    else:
        gamma_ratio = gamma((n-1)/2) / gamma(n / 2)
    C_n = np.sqrt((n - 1)/2) * gamma_ratio
    # Return adjusted value
    sighat *= C_n
    return sighat


###############################
# --- KURTOSIS ADJUSTMENT --- #

def sd_kappa(
            x: np.ndarray, 
            axis: int | None = None,
            ddof:int = 1, 
            debias_cumulants: bool = True,
            kurtosis_fun: Callable | None = None,
            **kwargs
            ) -> np.ndarray:
    """
    Adjust the vanilla SD estimator with the first-order adjustment from a Maclaurin expansion: sigma*[1-(kappa-1+2(n-1))/(8n)]^{-1}. See Giles (2021): http://web.uvic.ca/~dgiles/downloads/working_papers/std_dev.pdf

    Args
    ====
    x : np.ndarray
        An array or array-like object of arbitrary dimension: x.shape = (d1, d2, ..., dk)
    axis : int | None, optional
        Axis to calculate the SD over
    ddof : int, optional
        The degrees of freedom for the sample SD
    debias_cumulants : bool, optional
        Should the "debiased" versions of the cumulants to be used? See https://mathworld.wolfram.com/Cumulant.html (this is the default in pandas.kurtosis btw)
    kurtosis_fun : Callable | None, optional
        Should an alternative kurtosis calculation be used? Will pass in kwargs & axis.
    """
    # Input checks
    x, sighat, n = check_x_axis_ddof(x, axis, ddof)
    # Calculate kurtosis (kappa)
    if kurtosis_fun:
        assert isinstance(kurtosis_fun, callable), 'if you supply an other_method, it must be callabte'
        kappa = kurtosis_fun(x, axis=axis, **kwargs)
    if debias_cumulants:
        kappa = kurtosis(a=x, bias=False, **kwargs)
    else:
        kappa = kurtosis(a=x, bias=True, **kwargs)
    # Calculate C_n
    C_n_kappa = 1 / ( 1 - (kappa - 1 + 2/(n-1)) / (8*n) ) 
    # Return adjusted value
    sighat *= C_n_kappa
    return sighat


################################
# --- JACKKNIFE ADJUSTMENT --- #

def sd_jackknife(
        x: np.ndarray, 
        axis: int = 0,
        ddof: int = 0,
        **kwargs,
        ) -> np.ndarray:
    """
    Calculates the jackknife-adjusted sample SD by calculating the (LOO) bias, where the final estimator is: n*sighat - (n-1)*mean(sighat_loo)

    Args
    ====
    x : np.ndarray
        An array or array-like object of arbitrary dimension: x.shape = (d1, d2, ..., dk)
    axis : int, optional
        Axis to calculate the SD over
    ddof : int, optional
        The degrees of freedom for the sample SD
    """
    # Input checks
    x, sighat, n = check_x_axis_ddof(x, axis, ddof)
    # Calculat the LOO sample SD
    xbar = np.mean(x, axis = axis)
    xbar_loo = (n*xbar - x) / (n-1)
    mu_x2 = np.mean(x ** 2, axis=axis)
    # Unadjuasted LOO variance
    sigma2_loo = (n / (n - 1)) * (mu_x2 - x**2 / n - (n - 1) * xbar_loo**2 / n)
    # Apply DOF adjustment, if any
    n_adj = (n-1) / (n - ddof - 1)
    # Clip is needed for ≈ 0 values which are negative
    sighat_loo = np.sqrt(n_adj * np.clip(sigma2_loo, 0, None))
    sighat_loo_mu = np.mean(sighat_loo, axis=axis)
    # Calculate the bias and offset
    bias_jackknife = (n - 1) * (sighat_loo_mu - sighat)
    sighat -= bias_jackknife
    return sighat


################################
# --- BOOTSTRAP ADJUSTMENT --- #


def sd_bootstrap(
        x:np.ndarray, 
        axis: int = 0,
        ddof: int = 0,
        num_boot: int = 1000,
        random_state: int | None = None,
        **kwargs,
        ) -> np.ndarray:
    """
    Generates {num_boot} bootstrap replicates for a 1-d array

    Args
    ====
    x : np.ndarray
        An array or array-like object of arbitrary dimension: x.shape = (d1, d2, ..., dk)
    axis : int | None, optional
        Axis to calculate the SD over
    ddof : int, optional
        The degrees of freedom for the sample SD
    num_boot : int, optional
        If method=='bootstrap', how many bootstrap sample to draw? Note that this approach will broadcast the original array with an addition {num_boot} rows in the axis=-1 dimension, so keep that in mind for memorary consideration
    random_state : int | None, optional
        Reproducability seed for the bootstrap method
    """
    # Input checks
    x, sighat, n = check_x_axis_ddof(x, axis, ddof)
    num_dim = len(x.shape)
    # Generate {num_boot} sample SDs
    np.random.seed(random_state)
    if num_dim == 1:  # Fast to do random choice 
        idx = np.random.randint(low=0, high=n, size=n*num_boot)
        sighat_star = x[idx].reshape([n, num_boot]).std(ddof=ddof, axis=axis)
    else:  # Otherwise need to take along axis of index
        idx = np.random.randint(low=0, high=n, size=x.shape+(num_boot,))
        sighat_star = np.take_along_axis(np.expand_dims(x, -1), idx, axis=axis).\
                        std(ddof=ddof, axis=axis)
    # Calculate the bootstrapped means  
    sighat_star_mu = np.mean(sighat_star, axis=-1)
    # Calculate bias and offset
    bias_bs = sighat_star_mu - sighat
    sighat -= bias_bs
    return sighat

