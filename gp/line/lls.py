#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["least_squares"]

import numpy as np


def least_squares(x, y, yerr):
    # Build the design matrix.
    A = np.vander(x, 2)
    AT = A.T

    # Compute the mean and covariance of the posterior constraint.
    cov = np.linalg.inv(np.dot(AT, A / yerr[:, None] ** 2))
    mu = np.dot(cov, np.dot(AT, y / yerr ** 2))

    return mu, cov
