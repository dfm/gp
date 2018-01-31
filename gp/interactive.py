#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["setup_likelihood_sampler", "setup_conditional_sampler"]

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve

from ipywidgets import interact, interactive

from .data import true_parameters
from .transit_model import SillyTransitModel


def setup_likelihood_sampler(kernel):
    # Pre-compute some stuff.
    x0 = np.linspace(-20, 20, 100)
    r = np.abs(x0[:, None] - x0[None, :])

    # This function samples from the likelihood distribution.
    def sample_likelihood(amp, ell):
        rng = np.random.RandomState(123)
        K = kernel([amp, ell], r)
        K[np.diag_indices_from(K)] += 1e-9
        y0 = rng.multivariate_normal(np.zeros_like(x0), K, 6)
        plt.plot(x0, y0.T, "k", alpha=0.5)
        plt.ylim(-800, 800)
        plt.xlim(-20, 20)
        plt.show()

    w = interact(sample_likelihood, amp=(10, 500.0), ell=(1.0, 10.0))
    return w


def setup_conditional_sampler(x, y, yerr, kernel):
    # Pre-compute a bunch of things for speed.
    xs = np.linspace(-20, 20, 100)
    rxx = np.abs(x[:, None] - x[None, :])
    rss = np.abs(xs[:, None] - xs[None, :])
    rxs = np.abs(x[None, :] - xs[:, None])
    ye2 = yerr ** 2

    model = SillyTransitModel(*true_parameters)

    # This function samples from the conditional distribution and
    # plots those samples.
    def sample_conditional(amp, ell,
                           depth=np.exp(true_parameters[0]),
                           duration=np.exp(true_parameters[1]),
                           time=true_parameters[2]):
        rng = np.random.RandomState(123)

        # Compute the covariance matrices.
        Kxx = kernel([amp, ell], rxx)
        Kxx[np.diag_indices_from(Kxx)] += ye2
        Kss = kernel([amp, ell], rss)
        Kxs = kernel([amp, ell], rxs)

        # Compute the residuals.
        model.set_parameter_vector([np.log(depth), np.log(duration), time])
        resid = y - model.get_value(x)

        # Compute the likelihood.
        factor = cho_factor(Kxx, overwrite_a=True)
        ld = 2*np.sum(np.log(np.diag(factor[0])))
        a = cho_solve(factor, resid)
        ll = -0.5 * (np.dot(resid, a) + ld)

        # Compute the predictive mean.
        model_xs = model.get_value(xs)
        mu = np.dot(Kxs, a) + model_xs

        # Compute the predictive covariance.
        cov = Kss - np.dot(Kxs, np.linalg.solve(Kxx, Kxs.T))
        cov[np.diag_indices_from(cov)] += 1e-9

        # Sample and display the results.
        y0 = rng.multivariate_normal(mu, cov, 6)
        plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
        plt.plot(xs, y0.T, "k", alpha=0.5)
        plt.plot(xs, model_xs, ":r")
        plt.ylim(-250, 100)
        plt.xlim(-20, 20)
        plt.title("lnlike = {0}".format(ll))
        plt.show()

    w = interactive(sample_conditional,
                    amp=(10.0, 500.0), ell=(0.5, 10.0),
                    depth=(10.0, 500.0),
                    duration=(0.1, 2.0),
                    time=(-5.0, 5.0))
    return w
