"""
Main module.

https://github.com/audreyfeldroy/cookiecutter-pypackage

https://github.com/cookiecutter/cookiecutter?tab=readme-ov-file

https://simonwillison.net/2024/Jan/16/python-lib-pypi/
"""

# External modules
import numpy as np
# Internal modules
from .utils import \
                sd_jackknife, \
                    sd_bootstrap, \
                        sd_gaussian, \
                            sd_kappa 

# Valid methods
valid_std_methods = [
    'vanilla', 
    'jackknife', 
    'bootstrap', 
    'gaussian', 
    'kappa',
    ]


def std(
        x: np.ndarray,
        method: str,
        axis: int | None = None,
        ddof: int = 1,
        num_boot: int = 1000,
        random_state: int | None = None,
        **kwargs
        ) -> np.ndarray:
    """
    Main standard deviation adjustment method to debias or reduce the bias. 
    
    If :math:`\sigma^2 = E[X - E[X]]^2`, then we are looking for an estimator, :math:`S`, with the property :math:`E[S] = \sigma`. When :math:`S` is the sample standard deviation (SD), then :math:`E[S] \leq \sigma`, and we either adjust it with a scaling factor: :math:`E[S] \cdot C_n = \sigma`, or a non-parametric bias shift: :math:`E[S + \text{bias}(X)] \approx \sigma`.

    Parameters
    ----------
    x : np.ndarray
        An array or array-like object of arbitrary dimension: x.shape = (d1, d2, ..., dk)
    method: str
        Which method should be used? Must be one of:
            vanilla: No adjustment
            jackknife: Leave-one-out jackknife
            bootstrap: Bootstrap
            gaussian: Known Gaussian C_n calculation
            kappa: First-order adjustment based on kurtosis
    axis : int | None = None
        Axis to calculate the SD over
    ddof : int, optional
        The degrees of freedom for the sample SD; should be kept to 1
    num_boot : int, optional
        If method=='bootstrap', how many bootstrap sample to draw? Note that this approach will broadcast the original array with an addition {num_boot} rows in the axis=-1 dimension, so keep that in mind for memorary consideration
    random_state: int | None, optional
        Reproducability seed for the bootstrap method
    **kwargs
        Optional arguments to pass into methods, see utils.sd_{method} for additional details
    
    Returns
    -------
    np.ndarray
        If x.shape = (d1, d2, ..., dk), and axis=j, then returns a (d1, ..., dj-1, dj+1, ..., dk) array
    """
    # Input checks
    assert method in valid_std_methods, f'method must be one of {valid_std_methods}'
    # calculate the square root of the variance
    if method == 'jackknife':
        sighat = sd_jackknife(x, axis=axis, ddof=ddof, **kwargs)
    elif method == 'bootstrap':
        sighat = sd_bootstrap(x, axis=axis, ddof=ddof, 
                          num_boot=num_boot, 
                          random_state=random_state, 
                          **kwargs)
    elif method == 'kappa':
        sighat = sd_kappa(x, axis=axis, ddof=ddof, **kwargs)
    elif method == 'gaussian':
        sighat = sd_gaussian(x, axis=axis, ddof=ddof, **kwargs)
    else: # method == 'vanilla'
        sighat = np.std(x, axis=axis, ddof=ddof, **kwargs)
    return sighat
        

