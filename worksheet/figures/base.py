#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

# Monkey patch the path.
import os
import sys
d = os.path.dirname
base = d(os.path.abspath(__file__))
p = d(d(base))
sys.path.insert(0, p)
import gp
assert gp.__file__ == os.path.join(p, "gp", "__init__.pyc")


# Set up savefig.
def savefig(fig, fn):
    fig.savefig(os.path.join(base, fn))
