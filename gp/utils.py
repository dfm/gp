#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["DATA_DIR", "load_data"]

import os
import numpy as np

d = os.path.dirname
DATA_DIR = os.path.join(d(d(os.path.abspath(__file__))), "data")


def load_data(fn):
    return np.loadtxt(os.path.join(DATA_DIR, fn), unpack=True)


def test_lnlike(llfn):
    np.random.seed(1234)
    for N in [10, 50, 100, 500]:
        r = np.random.randn(N)
        x = np.sort(np.random.rand(N))
        K = np.exp(-0.5 * ((x[:, None] - x[None, :])/0.1) ** 2)
        K[np.diag_indices_from(K)] += 1e-3
        assert np.allclose(llfn(r, K), _baseline_lnlike(r, K)), N


def _baseline_lnlike(r, K):
    s, ld = np.linalg.slogdet(K)
    return -0.5 * (np.dot(r, np.linalg.solve(K, r)) + ld)
