# coding: utf-8
"""LightGBM, Light Gradient Boosting Machine.

Contributors: https://github.com/microsoft/LightGBM/graphs/contributors.
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
    from .plotting import (plot_importance, plot_split_value_histogram, plot_metric,
                           plot_tree, create_tree_digraph)
except ImportError:
    pass


dir_path = os.path.dirname(os.path.realpath(__file__))

if os.path.isfile(os.path.join(dir_path, 'VERSION.txt')):
    with open(os.path.join(dir_path, 'VERSION.txt')) as version_file:
        __version__ = version_file.read().strip()

__all__ = ['Dataset', 'Booster',
           'train', 'cv',
           'LGBMModel', 'LGBMRegressor', 'LGBMClassifier', 'LGBMRanker',
           'print_evaluation', 'record_evaluation', 'reset_parameter', 'early_stopping',
           'plot_importance', 'plot_split_value_histogram', 'plot_metric', 'plot_tree', 'create_tree_digraph']
