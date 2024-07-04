#!/usr/bin/env python

"""
Tests for `debiased_sd` package.

Ways to run manually: python -m unittest discover tests
"""

# External modules
import unittest
import numpy as np
from scipy.stats import binom
# Internal modules
from src.debiased_sd.estimators import std, valid_std_methods
# Parameters
seed = 1234
p_seq = [0.25, 0.5, 0.75]
m = 50
dist_binom = binom(p = p_seq, n = m)
oracle_sd = np.sqrt(dist_binom.stats('v'))
n = 500
err_tol = 2 / np.sqrt(n)
d1 = 5
d2 = 4
sizes = (n, d1, d2, len(p_seq))
data = dist_binom.rvs(size=sizes, random_state=seed)


class TestDebiasedSD(unittest.TestCase):
    """Tests for `debiased_sd` package."""

    def setUp(self):
        """Set up test fixtures, if any."""
        self.data = data
        self.methods = valid_std_methods

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_sd_methods(self):
        """Test the different SD adjustment methods"""
        for method in self.methods:
            with self.subTest(method=method):
                # Calculate sample SD with adjustment
                sighat = std(self.data, axis=0, ddof=1, method=method)
                sighat_mu = np.mean(sighat, axis=tuple(range(len(sizes)-2)))
                # Check within close value to oracle
                np.testing.assert_allclose(sighat_mu, oracle_sd, atol=err_tol)
                bias = np.mean(sighat_mu - oracle_sd)
                print(f'method = {method}, bias={bias:.3f}')
                