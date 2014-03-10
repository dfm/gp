#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["LightCurve"]

import copy
import numpy as np
import scipy.optimize as op

import george
from george.kernels import RBFKernel


class LightCurve(object):

    def __init__(self, time, flux, ferr, quality=None):
        # Mask bad data.
        m = np.isfinite(time) * np.isfinite(flux) * np.isfinite(ferr)
        if quality is not None:
            m *= quality
        self.time = np.atleast_1d(time)[m]
        self.flux = np.atleast_1d(flux)[m]
        self.ferr = np.atleast_1d(ferr)[m]

        # Normalize by the median.
        mu = np.median(self.flux)
        self.flux /= mu
        self.ferr /= mu

    def split(self, ts, normalize=True):
        ts = np.concatenate([[-np.inf], np.sort(np.atleast_1d(ts)), [np.inf]])
        datasets = []
        for i, t0 in enumerate(ts[:-1]):
            m = (np.isfinite(self.time) * (self.time >= t0)
                 * (self.time < ts[i + 1]))
            if np.any(m):
                ds = copy.deepcopy(self)
                ds.time = ds.time[m]
                ds.flux = ds.flux[m]
                ds.ferr = ds.ferr[m]
                if normalize:
                    mu = np.median(ds.flux)
                    ds.flux /= mu
                    ds.ferr /= mu
                datasets.append(ds)

        return datasets

    def autosplit(self, ttol, max_length=None):
        dt = self.time[1:] - self.time[:-1]
        m = dt > ttol
        ts = 0.5 * (self.time[1:][m] + self.time[:-1][m])
        datasets = self.split(ts)
        if max_length is not None:
            while any([len(d.time) > max_length for d in datasets]):
                datasets = [[d] if len(d.time) <= max_length
                            else d.split([d.time[int(0.5*len(d.time))]])
                            for d in datasets]
                datasets = [d for ds in datasets for d in ds]

        return datasets

    def optimize_hyperparams(self, p0, N=3):
        ts = np.linspace(self.time.min(), self.time.max(), N+2)
        lcs = self.split(ts[1:-1])
        return np.median(map(lambda l: l._op(p0), lcs), axis=0)

    def _op(self, p0):
        results = op.minimize(nll, np.log(p0), method="L-BFGS-B",
                              args=(self.time, self.flux, self.ferr))
        return np.exp(results.x)


def nll(p, t, f, fe):
    a, s = np.exp(p)
    gp0 = george.GaussianProcess(a * RBFKernel(s), tol=1e-16, nleaf=20)
    gp0.compute(t, fe)
    return -gp0.lnlikelihood(f-1)
