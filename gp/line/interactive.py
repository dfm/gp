#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["setup_likelihood_sampler", "setup_conditional_sampler"]

import numpy as np
import matplotlib.pyplot as pl

from IPython.display import display
from IPython.html.widgets import interact, interactive

from .lls import least_squares


def setup_likelihood_sampler(kernel):
    # Pre-compute some stuff.
    x0 = np.linspace(-5, 5, 300)
    r = x0[:, None] - x0[None, :]

    # This function samples from the likelihood distribution.
    def sample_likelihood(amp, ell):
        np.random.seed(123)
        K = kernel([amp, ell], r)
        y0 = np.random.multivariate_normal(np.zeros_like(x0), K, 6)
        pl.plot(x0, y0.T, "k", alpha=0.5)
        pl.ylim(-1.5, 1.5)
        pl.xlim(-5, 5)

    return interact(sample_likelihood, amp=(1.0e-4, 1.0), ell=(0.01, 3.0))


def setup_conditional_sampler(x, y, yerr, kernel):
    # Pre-compute a bunch of things for speed.
    xs = np.linspace(-6, 6, 300)
    rxx = x[:, None] - x[None, :]
    rss = xs[:, None] - xs[None, :]
    rxs = x[None, :] - xs[:, None]
    ye2 = yerr ** 2

    # Initialize at the least squares position.
    mu, _ = least_squares(x, y, yerr)

    # This function samples from the conditional distribution and
    # plots those samples.
    def sample_conditional(amp, ell, m=mu[0], b=mu[1]):
        np.random.seed(123)

        # Compute the covariance matrices.
        Kxx = kernel([amp, ell], rxx)
        Kxx[np.diag_indices_from(Kxx)] += ye2
        Kss = kernel([amp, ell], rss)
        Kxs = kernel([amp, ell], rxs)

        # Compute the residuals.
        resid = y - (m*x+b)
        model = m*xs+b
        a = np.linalg.solve(Kxx, resid)

        # Compute the likelihood.
        s, ld = np.linalg.slogdet(Kxx)
        ll = -0.5 * (np.dot(resid, a) + ld + len(x)*np.log(2*np.pi))

        # Compute the predictive mean.
        mu = np.dot(Kxs, a) + model

        # Compute the predictive covariance.
        cov = Kss - np.dot(Kxs, np.linalg.solve(Kxx, Kxs.T))

        # Sample and display the results.
        y0 = np.random.multivariate_normal(mu, cov, 6)
        pl.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
        pl.plot(xs, y0.T, "k", alpha=0.5)
        pl.plot(xs, m*xs+b, ":r")
        pl.ylim(-3, 3)
        pl.xlim(-6, 6)
        pl.title("lnlike = {0}".format(ll))

    w = interactive(sample_conditional, amp=(1.0e-4, 2.0), ell=(0.01, 3.0),
                    m=(-1.0, 5.0), b=(-3.0, 3.0))
    display(w)

    return w
