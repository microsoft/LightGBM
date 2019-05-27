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
import warnings
from platform import system

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

# REMOVEME: remove warning after 2.3.0 version release
if system() == 'Darwin':
    warnings.warn("Starting from version 2.2.1, the library file in distribution wheels for macOS "
                  "is built by the Apple Clang (Xcode_8.3.3) compiler.\n"
                  "This means that in case of installing LightGBM from PyPI via the ``pip install lightgbm`` command, "
                  "you don't need to install the gcc compiler anymore.\n"
                  "Instead of that, you need to install the OpenMP library, "
                  "which is required for running LightGBM on the system with the Apple Clang compiler.\n"
                  "You can install the OpenMP library by the following command: ``brew install libomp``.", UserWarning)
