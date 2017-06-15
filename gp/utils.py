#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["test_log_like", "test_kernel"]

import numpy as np

def test_log_like(llfn):
    np.random.seed(1234)
    for N in [10, 50, 100, 500]:
        r = np.random.randn(N)
        x = np.sort(np.random.rand(N))
        K = np.exp(-0.5 * ((x[:, None] - x[None, :])/0.1) ** 2)
        K[np.diag_indices_from(K)] += 1e-3
        assert np.allclose(llfn(r, K), _baseline_lnlike(r, K)), N
    print(u"☺︎")


def _baseline_lnlike(r, K):
    s, ld = np.linalg.slogdet(K)
    return -0.5 * (np.dot(r, np.linalg.solve(K, r)) + ld)


def test_kernel(kernelfn):
    np.random.seed(1234)
    for N in [10, 50, 100, 200]:
        x = np.sort(np.random.rand(N))
        params = np.random.rand(2)
        k1 = kernelfn(params, x[:, None] - x[None, :])
        k2 = _baseline_kernel(params, x[:, None] - x[None, :])
        assert np.allclose(k1, k2), N
    print(u"☺︎")


def _baseline_kernel(params, dx):
    return params[0] * params[0] * np.exp(-0.5 * (dx / params[1]) ** 2)
