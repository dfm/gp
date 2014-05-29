#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["fit_independent"]

import os
import numpy as np
import matplotlib.pyplot as pl
from utils import load_data


def fit_independent(rng=(-5, 5)):
    x, y, yerr = load_data("line_data.txt")
    true_m, true_b, _, _ = load_data("line_true_params.txt")

    # Build the design matrix.
    A = np.vander(x, 2)
    AT = A.T

    # Compute the mean and covariance of the posterior constraint.
    cov = np.linalg.inv(np.dot(AT, A / yerr[:, None] ** 2))
    mu = np.dot(cov, np.dot(AT, y / yerr ** 2))

    # Numerically compute the constraints in the data coordinates.
    x0 = np.linspace(rng[0], rng[1], 5000)
    samples = np.dot(np.vander(x0, 2),
                     np.random.multivariate_normal(mu, cov, 1000).T)
    mean = np.mean(samples, axis=1)
    std = np.std(samples, axis=1)

    # Plot these constraints with the truth and the data points.
    fig = pl.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
    ax.fill_between(x0, mean+std, mean-std, color="r", alpha=0.3)
    ax.plot(x0, true_m*x0+true_b, "k")
    ax.set_xlim(rng)
    ax.set_ylim(np.array([-1, 1]) * max(np.abs(ax.get_ylim())))
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title("assuming independent uncertainties")
    fig.savefig(os.path.join("figures", "line_independent.png"))

if __name__ == "__main__":
    fit_independent()
