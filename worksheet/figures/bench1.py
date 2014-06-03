#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
import gp
import numpy as np
import matplotlib.pyplot as pl


# Define the Gaussian ln-likelihood.
def lnlike(r, K):
    s, ld = np.linalg.slogdet(K)
    return -0.5 * (np.dot(r, np.linalg.solve(K, r)) + ld)
gp.utils.test_lnlike(lnlike)


# Define the exponential squared kernel.
def kernel(params, dx):
    return params[0] * params[0] * np.exp(-0.5 * (dx / params[1]) ** 2)
gp.utils.test_kernel(kernel)

# Run the benchmark.
hyper = [1.0, 0.1]
gp.benchmark("basic", kernel, lnlike, hyper)
