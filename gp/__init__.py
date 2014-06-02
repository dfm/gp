#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ = ["line", "utils"]

# Support X-windows.
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as pl
pl.ion()

from . import line
from . import utils
from .benchmarking import benchmark
