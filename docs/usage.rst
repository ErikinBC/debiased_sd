=====
Usage
=====

Example of how to call in the debiased sample SD estimator::

    # Load modules
    import numpy as np
    from scipy.stats import norm
    from debiased_sd import estimators
    from debiased_sd import utils
    print(f'The following are valid sample SD methods = {estimators.valid_std_methods}')
    
    # You can learn more about each method by running: `help('debiased_sd.utils.{std_method}')
    help('debiased_sd.utils.sd_jackknife')
    
    # Data of size (n, d1, d2, ..., dk), where n is the sample size
    n, d1, d2, d3 = 10, 5, 4, 3
    data = norm().rvs(size = (n, d1, d2, d3), random_state=1)
    
    # Calculate the jackknife method
    sighat_vanilla = np.std(data, axis=0, ddof=1)
    sighat_jackknife = estimators.std(data, axis=0, method='jackknife')
    
    # Jackknife helps to debias to get sample SD closer to population parameter of 1
    assert np.all(sighat_jackknife > sighat_vanilla)
    print(f'Sample SD mean: Jackknife = {sighat_jackknife.mean():.2f}, Vanilla = {sighat_vanilla.mean():.2f}')
    