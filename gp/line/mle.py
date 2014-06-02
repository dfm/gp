#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["fit_mle"]

import numpy as np
import scipy.optimize as op


def fit_mle(x, y, yerr, m0, b0, hyper0, kernel, lnlike):
    # Define the negative ln-likelihood.
    def nll(params):
        K = kernel(params[2:], dx)
        K[np.diag_indices_from(K)] += ye2
        r = y - (params[0] * x + params[1])
        return -lnlike(r, K)
    dx = x[None, :] - x[:, None]
    ye2 = yerr ** 2

    # Maximize (minimize) the (negative) ln-likelihood.
    bounds = [(None, None), (None, None)] + [(1e-10, None)] * len(hyper0)
    initial = np.concatenate(([m0, b0], hyper0))
    results = op.minimize(nll, initial, method="L-BFGS-B", bounds=bounds)

    print(results)
    return results.x
