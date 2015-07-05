#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["benchmark"]

import time
import numpy as np
import matplotlib.pyplot as pl

try:
    from itertools import izip
except ImportError:
    izip = zip


def benchmark(kernelfn, lnlikefn, params, N=2 ** np.arange(8, 12)):
    # Add a data type to the timing array.
    dt = [
        ("kernel", np.float64),
        ("lnlike", np.float64),
        ("total", np.float64),
    ]
    t = np.empty(len(N), dtype=dt)

    for i, n in enumerate(N):
        # Generate some fake data.
        x = n * np.sort(np.random.rand(n)) / 64.
        y = np.sin(x)

        # Build the kernel.
        strt = time.time()
        K = kernelfn(params, x[:, None] - x[None, :])
        K[np.diag_indices_from(K)] += 1e-10

        # Save the kernel build time.
        t["kernel"][i] = time.time() - strt

        # Compute the likelihood.
        mid = time.time()
        lnlikefn(y, K)

        # Save the likelihood computation time and the total time.
        t["lnlike"][i] = time.time() - mid
        t["total"][i] = time.time() - strt
        print(n, t["total"][i])

    # Compute scaling and plot the results.
    fig = pl.figure()
    ax = fig.add_subplot(111)
    for i, ((l, _), c) in enumerate(izip(dt, "rgb")):
        x, y = np.log10(N), np.log10(t[l])
        p = np.polyfit(x, y, 1)
        ax.plot(x, y, "o", label="{0} [{1:.2f}]".format(l, p[0]), color=c)
        ax.plot(x, np.polyval(p, x), color=c)
    ax.set_xlabel("log10(time)")
    ax.set_ylabel("log10(N)")
    ax.legend(loc="lower right")

    return N, t
