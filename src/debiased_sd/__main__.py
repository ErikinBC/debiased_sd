def main():
    import numpy as np
    from .estimators import std, valid_std_methods
    # Generate some small-sample data
    nsim = 1000
    nsample = 8
    np.random.seed(nsim)
    x = np.random.randn(nsample, nsim)
    di_sighat = dict.fromkeys(valid_std_methods)
    for method in valid_std_methods:
        di_sighat[method] = std(x, method=method, axis=0)
    # Expect vanilla to be negatively biased
    di_bias = {k:v.mean() -1 for k,v in di_sighat.items()}
    assert di_bias['vanilla'] < 0, 'expected a negative bias'
    del di_bias['vanilla']
    bias_rest = np.array(list(di_bias.values()))
    assert np.all(bias_rest > 0), f'Expected the other estimators to be conservative, not {bias_rest.round(2)}'
    print("The debiased_sd package has installed correctly")

if __name__ == "__main__":
    main()