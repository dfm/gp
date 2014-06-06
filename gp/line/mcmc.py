#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["sample"]

import emcee
import numpy as np


def sample(x, y, yerr, user_kernel, user_lnlike, initial,
           steps=5000, burnin=5000, thin=37):
    ye2 = yerr ** 2
    dx = x[:, None] - x[None, :]

    # Define the probabilistic model.
    def lnprior(theta):
        theta = np.atleast_1d(theta)
        if not np.all((-4.0 < theta[:2]) * (theta[:2] < 6.0)):
            return -np.inf
        if not np.all((-3.0 < theta[2:]) * (theta[2:] < 2.0)):
            return -np.inf
        return 0.0
        # return np.sum(theta[2:])

    def lnlike(theta):
        m, b = theta[:2]
        K = user_kernel(np.exp(theta[2:]), dx)
        K[np.diag_indices_from(K)] += ye2
        return user_lnlike(y - (m*x+b), K)

    def lnprob(theta):
        lp = lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = lnlike(theta)
        if not np.isfinite(ll):
            return -np.inf
        return ll + lp

    # Set up the sampler.
    ndim, nwalkers = len(initial), 32
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)

    # Initialize the walkers.
    initial = np.atleast_1d(initial)
    initial[2:] = np.log(initial[2:])
    p0 = [initial + 1e-8 * np.random.randn(ndim) for i in range(nwalkers)]
    pos, lp, state = sampler.run_mcmc(p0, burnin)
    sampler.reset()
    sampler.run_mcmc(pos, steps)

    return sampler.flatchain[::thin, :]
