# coding: utf-8
"""LightGBM, Light Gradient Boosting Machine.

Contributors: https://github.com/Microsoft/LightGBM/graphs/contributors
"""
from __future__ import absolute_import

from .basic import Booster, Dataset
from .callback import (early_stopping, print_evaluation, record_evaluation,
                       reset_parameter)
from .engine import cv, train
import os

try:
    from .sklearn import LGBMModel, LGBMRegressor, LGBMClassifier, LGBMRanker
except ImportError:
    pass
try:
    from .plotting import plot_importance, plot_metric, plot_tree, create_tree_digraph
except ImportError:
    pass

version = '2.0.1'
if os.path.isfile('./VERSION.txt'):
    version = open('./VERSION.txt').read().strip()

__version__ = version

__all__ = ['Dataset', 'Booster',
           'train', 'cv',
           'LGBMModel', 'LGBMRegressor', 'LGBMClassifier', 'LGBMRanker',
           'print_evaluation', 'record_evaluation', 'reset_parameter', 'early_stopping',
           'plot_importance', 'plot_metric', 'plot_tree', 'create_tree_digraph']
