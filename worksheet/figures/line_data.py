#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
from base import savefig
import gp

x, y, yerr = gp.line.load_data()
fig = gp.line.plot_data(x, y, yerr)
savefig(fig, "line_data.pdf")
