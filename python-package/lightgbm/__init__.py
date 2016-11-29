# coding: utf-8
"""LightGBM, Light Gradient Boosting Machine.

Contributors: https://github.com/Microsoft/LightGBM/graphs/contributors
"""

from __future__ import absolute_import

import os

from .basic import Predictor, Dataset, Booster
from .engine import train, cv
try:
    from .sklearn import LGBMModel, LGBMRegressor, LGBMClassifier
except ImportError:
    pass

VERSION_FILE = os.path.join(os.path.dirname(__file__), 'VERSION')
with open(VERSION_FILE) as f:
    __version__ = f.read().strip()

__all__ = ['Dataset', 'Booster',
           'train', 'cv',
           'LGBMModel','LGBMRegressor', 'LGBMClassifier']