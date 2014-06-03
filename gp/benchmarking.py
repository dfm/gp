#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["benchmark"]

import time
import numpy as np
import matplotlib.pyplot as pl


def benchmark(name, kernelfn, lnlikefn, params, N=2 ** np.arange(8, 12)):
    print("Running benchmark: '{0}'".format(name))
    t = np.empty((len(N), 3))
    for i, n in enumerate(N):
        # Generate some fake data.
        x = n * np.sort(np.random.rand(n)) / 64.
        y = np.sin(x)

        # Build the kernel.
        strt = time.time()
        K = kernelfn(params, x[:, None] - x[None, :])
        K[np.diag_indices_from(K)] += 1e-10

        # Save the kernel build time.
        t[i, 0] = time.time() - strt

        # Compute the likelihood.
        mid = time.time()
        lnlikefn(y, K)

        # Save the likelihood computation time and the total time.
        t[i, 1] = time.time() - mid
        t[i, 2] = time.time() - strt

        print(n, t[i])

    # Save the results.
    fn = "benchmark-{0}.txt".format(name)
    print("Saving results to '{0}'".format(fn))
    with open(fn, "w") as f:
        f.write("# N kernel lnlike total\n")
        for i, n in enumerate(N):
            f.write("{0} {1}\n".format(n, " ".join(map("{0}".format, t[i]))))

    # Compute scaling and plot the results.
    fig = pl.figure()
    ax = fig.add_subplot(111)
    for i, (l, c) in enumerate([("kernel", "r"),
                                ("lnlike", "g"),
                                ("total", "b")]):
        x, y = np.log10(N), np.log10(t[:, i])
        p = np.polyfit(x, y, 1)
        ax.plot(x, y, "o", label="{0} [{1:.2f}]".format(l, p[0]), color=c)
        ax.plot(x, np.polyval(p, x), color=c)
    ax.set_title(name)
    ax.set_xlabel("log10(time)")
    ax.set_ylabel("log10(N)")
    ax.legend(loc="lower right")

    plot_fn = "benchmark-{0}.pdf".format(name)
    print("Saving plot to '{0}'".format(plot_fn))
    fig.savefig(plot_fn)

    return N, t
