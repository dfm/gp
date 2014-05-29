#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = []

import os
import emcee
import numpy as np
from scipy.linalg import cho_factor, cho_solve

from utils import load_data, plot_results

# Load the data and the true line parameters.
x, y, yerr = load_data("line_data.txt")
true_m, true_b, true_alpha, true_ell = load_data("line_true_params.txt")


def kernel_function(r, hyperpars):
    a2, l2 = np.exp(2 * hyperpars)
    return a2 * np.exp(-0.5 * r**2 / l2)


def build_covariance_matrix(x, yerr, hyperpars):
    C = kernel_function(x[:, None] - x[None, :], hyperpars)
    C[np.diag_indices_from(C)] += yerr ** 2
    return C


def lnlike_base((m, b), x, y, (factor, flag)):
    r = y - (m*x + b)
    return -0.5 * np.dot(r, cho_solve((factor, flag), r))


def lnprior_base((m, b)):
    if not -5 < m < 5:
        return -np.inf
    if not -5 < b < 5:
        return -np.inf
    return 0.0


def lnlike(p, x, y, yerr):
    # Compute the covariance matrix and its Cholesky factorization.
    C = build_covariance_matrix(x, yerr, p[2:])
    factor, flag = cho_factor(C, overwrite_a=True)

    # Use this factorization to compute the ln-determinant.
    lndet = np.sum(2*np.log(np.diag(factor)))

    # Compute the ln-likelihood using this factorization.
    ll = lnlike_base(p[:2], x, y, (factor, flag))
    ll -= 0.5 * lndet

    return ll


def lnprior(p):
    lp = lnprior_base(p[:2])
    if not np.isfinite(lp):
        return -np.inf
    if np.any(np.abs(p[2:]) > 5):
        return -np.inf
    return lp + 0.0


def lnprob(p, x, y, yerr):
    lp = lnprior(p)
    if not np.isfinite(lp):
        return -np.inf
    return lnlike(p, x, y, yerr) + lp


def fit_emcee(rng=(-5, 5)):
    # Initialize the walkers.
    ndim, nwalkers = 4, 32
    pos = [np.random.randn(ndim) for i in xrange(nwalkers)]

    # Initialize the sampler.
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr))

    # Run a burn-in.
    print("Running burn-in")
    pos, lp, state = sampler.run_mcmc(pos, 1000)
    sampler.reset()

    # Run the production chain.
    print("Running production")
    sampler.run_mcmc(pos, 500)
    print("Done")

    fig = plot_results(x, y, yerr, sampler.flatchain, truth=(true_m, true_b))
    fig.savefig(os.path.join("figures", "line_emcee.png"))

if __name__ == "__main__":
    fit_emcee()
