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
