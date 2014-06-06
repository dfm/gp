#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["plot_truth", "plot_data", "plot_results"]

import numpy as np
import matplotlib.pyplot as pl

from .. import triangle
from ..utils import load_data


def plot_truth(fig=None):
    fig, ax = _get_fig_ax(fig)

    # Load the true parameters the were used to generate the line.
    m, b, _, _ = load_data("line_true_params.txt")

    # Plot the true line
    x0 = np.array([-5, 5])
    ax.plot(x0, m*x0+b, "k")

    return fig


def plot_data(x, y, yerr, fig=None, truth=True):
    fig, ax = _get_fig_ax(fig)

    # Plot the true line if requested.
    if truth:
        plot_truth(fig=fig)

    # Plot the error bars.
    ax.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)

    # Format the axes.
    _format_axes(ax)

    return fig


def plot_results(x, y, yerr, samples, truth=True, color="r", data_fig=None):
    if data_fig is None:
        # Plot the data.
        data_fig = plot_data(x, y, yerr, truth=truth)
        data_fig, data_ax = _get_fig_ax(data_fig)
    else:
        data_ax = data_fig.gca()

    # Generate the constraints in data space.
    x0 = np.linspace(-5, 5, 500)
    samples = np.atleast_1d(samples)
    if len(samples.shape) == 2:
        lines = np.dot(np.vander(x0, 2), samples[:, :2].T)
        mean = np.mean(lines, axis=1)
        std = np.std(lines, axis=1)
        data_ax.fill_between(x0, mean+std, mean-std, color=color, alpha=0.3)
    else:
        data_ax.plot(x0, np.dot(np.vander(x0, 2), samples[:2]), color=color)

    # Plot the triangle plot.
    triangle_fig = triangle.corner(samples, bins=24,
                                   labels=["m", "b", "alpha", "ell"],
                                   truths=load_data("line_true_params.txt"))

    _format_axes(data_ax)
    return data_fig, triangle_fig


def _get_fig_ax(fig):
    # Allow over-plotting.
    if fig is None:
        fig = pl.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
    else:
        ax = fig.gca()
    return fig, ax


def _format_axes(ax, rng=(-5, 5)):
    ax.set_xlim(rng)
    ax.set_ylim(np.array([-1, 1]) * max(np.abs(ax.get_ylim())))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return ax
