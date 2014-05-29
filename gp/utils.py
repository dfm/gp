#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["DATA_DIR", "load_data"]

import os
import numpy as np
import matplotlib.pyplot as pl

d = os.path.dirname
DATA_DIR = os.path.join(d(d(os.path.abspath(__file__))), "data")


def load_data(fn):
    return np.loadtxt(os.path.join(DATA_DIR, fn), unpack=True)


def plot_results(x, y, yerr, samples, rng=(-5, 5), truth=None, fig=None,
                 color="r"):
    x0 = np.linspace(rng[0], rng[1], 500)
    lines = np.dot(np.vander(x0, 2), samples[:, :2].T)
    mean = np.mean(lines, axis=1)
    std = np.std(lines, axis=1)

    if fig is None:
        fig = pl.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
    else:
        ax = fig.gca()

    ax.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
    ax.fill_between(x0, mean+std, mean-std, color=color, alpha=0.3)
    if truth is not None:
        ax.plot(x0, truth[0]*x0+truth[1], "k")
    ax.set_xlim(rng)
    ax.set_ylim(np.array([-1, 1]) * max(np.abs(ax.get_ylim())))
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")

    return fig
