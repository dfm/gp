#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
from base import savefig
import gp
import numpy as np

x, y, yerr = gp.line.load_data()

# Run the least-squares model.
mu, cov = gp.line.least_squares(x, y, yerr)
print("Mean: {0}\nCovariance:\n{1}".format(mu, cov))

# Generate some posterior samples.
samples = np.random.multivariate_normal(mu, cov, 20000)

# Plot the results.
data_fig, triangle_fig = gp.line.plot_results(x, y, yerr, samples)

savefig(data_fig, "line_lls.pdf")
savefig(triangle_fig, "line_lls_triangle.pdf")
