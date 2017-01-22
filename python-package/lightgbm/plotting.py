# coding: utf-8
# pylint: disable = C0103
"""Plotting Library."""
from __future__ import absolute_import

import numpy as np

from .basic import Booster, is_numpy_1d_array
from .sklearn import LGBMModel


def plot_importance(booster, ax=None, height=0.2,
                    xlim=None, ylim=None, title='Feature importance',
                    xlabel='Feature importance', ylabel='Features',
                    importance_type='split', max_num_features=None,
                    ignore_zero=True, grid=True, **kwargs):
    """Plot model feature importances.

    Parameters
    ----------
    booster : Booster, LGBMModel or array
        Booster or LGBMModel instance, or array of feature importances
    ax : matplotlib Axes
        Target axes instance. If None, new figure and axes will be created.
    height : float
        Bar height, passed to ax.barh()
    xlim : tuple
        Tuple passed to axes.xlim()
    ylim : tuple
        Tuple passed to axes.ylim()
    title : str
        Axes title. Pass None to disable.
    xlabel : str
        X axis title label. Pass None to disable.
    ylabel : str
        Y axis title label. Pass None to disable.
    importance_type : str
        How the importance is calculated: "split" or "gain"
        "split" is the number of times a feature is used in a model
        "gain" is the total gain of splits which use the feature
    max_num_features : int
        Max number of top features displayed on plot.
        If None or smaller than 1, all features will be displayed.
    ignore_zero : bool
        Ignore features with zero importance
    grid : bool
        Whether add grid for axes
    **kwargs :
        Other keywords passed to ax.barh()

    Returns
    -------
    ax : matplotlib Axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError('You must install matplotlib for plotting library')

    if isinstance(booster, LGBMModel):
        importance = booster.booster_.feature_importance(importance_type=importance_type)
    elif isinstance(booster, Booster):
        importance = booster.feature_importance(importance_type=importance_type)
    elif is_numpy_1d_array(booster) or isinstance(booster, list):
        importance = booster
    else:
        raise ValueError('booster must be Booster or array instance')

    if not len(importance):
        raise ValueError('Booster feature_importances are empty')

    tuples = sorted(enumerate(importance), key=lambda x: x[1])
    if ignore_zero:
        tuples = [x for x in tuples if x[1] > 0]
    if max_num_features is not None and max_num_features > 0:
        tuples = tuples[-max_num_features:]
    labels, values = zip(*tuples)

    if ax is None:
        _, ax = plt.subplots(1, 1)

    ylocs = np.arange(len(values))
    ax.barh(ylocs, values, align='center', height=height, **kwargs)

    for x, y in zip(values, ylocs):
        ax.text(x + 1, y, x, va='center')

    ax.set_yticks(ylocs)
    ax.set_yticklabels(labels)

    if xlim is not None:
        if not isinstance(xlim, tuple) or len(xlim) != 2:
            raise ValueError('xlim must be a tuple of 2 elements')
    else:
        xlim = (0, max(values) * 1.1)
    ax.set_xlim(xlim)

    if ylim is not None:
        if not isinstance(ylim, tuple) or len(ylim) != 2:
            raise ValueError('ylim must be a tuple of 2 elements')
    else:
        ylim = (-1, len(values))
    ax.set_ylim(ylim)

    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.grid(grid)
    return ax
