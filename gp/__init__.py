#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ = ["line", "utils", "benchmark"]

import socket

if "rcc.psu.edu" in socket.gethostname():
    # Support X-windows on the PSU cluster. This is a total HACK.
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as pl
    pl.ion()

from . import line
from . import utils
from .benchmarking import benchmark
