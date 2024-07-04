def main():
    """
    Call with `python3 -m debiased_sd`.
    Note that if the package isn't install, you need to call python3 -m src.debiased_sd
    """
    # Load modules
    import numpy as np
    from .estimators import std, valid_std_methods
    # Generate some small-sample data
    nsim = 1000
    nsample = 8
    np.random.seed(nsim)
    x = np.random.randn(nsample, nsim)
    # Generate measure of bias
    di_sighat = dict.fromkeys(valid_std_methods)
    for method in valid_std_methods:
        di_sighat[method] = std(x, method=method, axis=0)
    di_bias = {k: v.mean() - 1 for k, v in di_sighat.items()}
    # Expect vanilla to be negatively biased
    assert di_bias['vanilla'] < 0, 'expected a negative bias'
    del di_bias['vanilla']
    print('Vanilla sample SD has negative bias')
    # Expect other methods to lean conservative
    bias_rest = np.array(list(di_bias.values()))
    assert np.all(bias_rest > 0), \
        f'Expected the other estimators to be conservative, not {bias_rest.round(2)}'
    print('Non-vanilla methods have positive bias')
    # Success!
    print("The debiased_sd package has installed correctly")


if __name__ == "__main__":
    main()
