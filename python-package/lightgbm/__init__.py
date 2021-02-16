# coding: utf-8
"""LightGBM, Light Gradient Boosting Machine.

Contributors: https://github.com/microsoft/LightGBM/graphs/contributors.
"""
import os

from .basic import Booster, Dataset, register_logger
from .callback import early_stopping, print_evaluation, record_evaluation, reset_parameter
from .engine import CVBooster, cv, train

try:
    from .sklearn import LGBMClassifier, LGBMModel, LGBMRanker, LGBMRegressor
except ImportError:
    pass
try:
    from .plotting import create_tree_digraph, plot_importance, plot_metric, plot_split_value_histogram, plot_tree
except ImportError:
    pass
try:
    from .dask import DaskLGBMClassifier, DaskLGBMRanker, DaskLGBMRegressor
except ImportError:
    pass


dir_path = os.path.dirname(os.path.realpath(__file__))

if os.path.isfile(os.path.join(dir_path, 'VERSION.txt')):
    with open(os.path.join(dir_path, 'VERSION.txt')) as version_file:
        __version__ = version_file.read().strip()

__all__ = ['Dataset', 'Booster', 'CVBooster',
           'register_logger',
           'train', 'cv',
           'LGBMModel', 'LGBMRegressor', 'LGBMClassifier', 'LGBMRanker',
           'DaskLGBMRegressor', 'DaskLGBMClassifier', 'DaskLGBMRanker',
           'print_evaluation', 'record_evaluation', 'reset_parameter', 'early_stopping',
           'plot_importance', 'plot_split_value_histogram', 'plot_metric', 'plot_tree', 'create_tree_digraph']
