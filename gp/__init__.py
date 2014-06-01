#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ = ["line"]

# Support X-windows.
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as pl
pl.ion()

from . import line
