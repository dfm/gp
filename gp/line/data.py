#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
from ..utils import load_data as base_load_data


def load_data():
    return base_load_data("line_data.txt")
