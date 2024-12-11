# coding: utf-8
"""LightGBM, Light Gradient Boosting Machine.

Contributors: https://github.com/microsoft/LightGBM/graphs/contributors.
"""

from pathlib import Path

# .basic is intentionally loaded as early as possible, to dlopen() lib_lightgbm.{dll,dylib,so}
# and its dependencies as early as possible
from .basic import Booster, Dataset, Sequence, register_logger
from .callback import EarlyStopException, early_stopping, log_evaluation, record_evaluation, reset_parameter
from .dask import DaskLGBMClassifier, DaskLGBMRanker, DaskLGBMRegressor
from .engine import CVBooster, cv, train
from .plotting import create_tree_digraph, plot_importance, plot_metric, plot_split_value_histogram, plot_tree
from .sklearn import LGBMClassifier, LGBMModel, LGBMRanker, LGBMRegressor

_version_path = Path(__file__).absolute().parent / "VERSION.txt"
if _version_path.is_file():
    __version__ = _version_path.read_text(encoding="utf-8").strip()

__all__ = [
    "Dataset",
    "Booster",
    "CVBooster",
    "Sequence",
    "register_logger",
    "train",
    "cv",
    "LGBMModel",
    "LGBMRegressor",
    "LGBMClassifier",
    "LGBMRanker",
    "DaskLGBMRegressor",
    "DaskLGBMClassifier",
    "DaskLGBMRanker",
    "log_evaluation",
    "record_evaluation",
    "reset_parameter",
    "early_stopping",
    "EarlyStopException",
    "plot_importance",
    "plot_split_value_histogram",
    "plot_metric",
    "plot_tree",
    "create_tree_digraph",
]
