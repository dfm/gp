#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["generate_line_data"]

import numpy as np
import matplotlib.pyplot as pl


def covariance_matrix(x, yerr, params):
    C = params[0]**2*np.exp(-0.5*(x[:, None]-x[None, :])**2 / params[1]**2)
    C[np.diag_indices_from(C)] += yerr**2
    return C


def generate_line_data(seed, N=50, rng=(-5, 5), yerrrng=(0.1, 0.5)):
    np.random.seed(seed)

    # Compute the y-values for the true line.
    true_m, true_b = 0.5, -0.25
    x = np.sort(np.random.uniform(rng[0], rng[1], N))
    true_y = true_m * x + true_b

    # Compute the covariance matrix.
    cov_params = [0.7, 1.3]
    yerr = np.random.uniform(yerrrng[0], yerrrng[1], N)
    C = covariance_matrix(x, yerr, cov_params)
    y = np.random.multivariate_normal(true_y, C)

    # Save the data.
    dt = [("x", np.float64), ("y", np.float64), ("yerr", np.float64)]
    data = np.array(zip(x, y, yerr), dtype=dt)
    np.savetxt("line_data.txt", data, header="x y yerr")
    np.savetxt("line_true_cov.txt", C,
               header=("The true covariance matrix for the data "
                       "(for reference)"))
    np.savetxt("line_true_params.txt", [[true_m, true_b] + cov_params],
               header=("The true parameters used to generate the data "
                       "(for reference)\nm b alpha ell"))

    # Plot the data.
    fig = pl.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
    ax.plot(x, true_y, "k")
    ax.set_xlim(rng)
    ax.set_ylim(np.array([-1, 1]) * max(np.abs(ax.get_ylim())))
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title("the data")
    fig.savefig("line_data.png")

    fig.clf()
    ax = fig.add_subplot(111)
    ax.imshow(C, cmap="gray", interpolation="nearest")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title("covariance matrix for the data")
    fig.savefig("line_cov.png")


if __name__ == "__main__":
    generate_line_data(123456)
