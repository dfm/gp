# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
from .modeling import Model

__all__ = ["SillyTransitModel"]


class SillyTransitModel(Model):
    parameter_names = ["log_depth", "log_duration", "time"]

    def get_value(self, t):
        chi2 = (t-self.time)**2*np.exp(-2*self.log_duration)
        return -np.exp(self.log_depth - 0.5*chi2)
