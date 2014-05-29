#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["fit_independent"]

import os
import numpy as np

from utils import load_data, plot_results


def fit_independent(rng=(-5, 5)):
    x, y, yerr = load_data("line_data.txt")
    true_m, true_b, _, _ = load_data("line_true_params.txt")

    # Build the design matrix.
    A = np.vander(x, 2)
    AT = A.T

    # Compute the mean and covariance of the posterior constraint.
    cov = np.linalg.inv(np.dot(AT, A / yerr[:, None] ** 2))
    mu = np.dot(cov, np.dot(AT, y / yerr ** 2))

    # Plot these constraints with the truth and the data points.
    samples = np.random.multivariate_normal(mu, cov, 1000)
    fig = plot_results(x, y, yerr, samples, truth=(true_m, true_b))
    fig.gca().set_title("assuming independent uncertainties")
    fig.savefig(os.path.join("figures", "line_independent.png"))

if __name__ == "__main__":
    fit_independent()
