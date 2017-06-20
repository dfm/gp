# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
from .transit_model import SillyTransitModel

__all__ = ["generate_dataset", "true_parameters"]

true_parameters = np.array([np.log(200.0), 0.0, 0.0])

def generate_dataset(N=12*10, seed=421, outliers=False):
    rng = np.random.RandomState(seed)
    x = np.sort(rng.uniform(-N / 6.0, N / 6.0, N))
    yerr = rng.uniform(10, 20, N)
    model = SillyTransitModel(log_depth=np.log(200.0),
                              log_duration=np.log(1.0),
                              time=0.0)
    model.set_parameter_vector(true_parameters)
    mean = model.get_value(x)

    K = 100.0**2 * np.exp(-0.5 * (x[:, None] - x[None, :])**2 / 8.0**2)
    K[np.diag_indices_from(K)] += yerr**2

    y = rng.multivariate_normal(mean, K)

    if outliers:
        m = rng.rand(N) < 0.1
        y[m] += outliers * rng.randn(m.sum())

    return x, y, yerr
