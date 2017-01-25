# coding: utf-8
# pylint: disable = C0103
"""Plotting Library."""
from __future__ import absolute_import

from io import BytesIO

import numpy as np

from .basic import Booster, is_numpy_1d_array
from .sklearn import LGBMModel


def check_not_tuple_of_2_elements(obj):
    """check object is not tuple or does not have 2 elements"""
    return not isinstance(obj, tuple) or len(obj) != 2


def plot_importance(booster, ax=None, height=0.2,
                    xlim=None, ylim=None, title='Feature importance',
                    xlabel='Feature importance', ylabel='Features',
                    importance_type='split', max_num_features=None,
                    ignore_zero=True, figsize=None, grid=True, **kwargs):
    """Plot model feature importances.

    Parameters
    ----------
    booster : Booster, LGBMModel or array
        Booster or LGBMModel instance, or array of feature importances
    ax : matplotlib Axes
        Target axes instance. If None, new figure and axes will be created.
    height : float
        Bar height, passed to ax.barh()
    xlim : tuple of 2 elements
        Tuple passed to axes.xlim()
    ylim : tuple of 2 elements
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
    figsize : tuple of 2 elements
        Figure size
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
        raise ImportError('You must install matplotlib to plot importance.')

    if isinstance(booster, LGBMModel):
        importance = booster.booster_.feature_importance(importance_type=importance_type)
    elif isinstance(booster, Booster):
        importance = booster.feature_importance(importance_type=importance_type)
    elif is_numpy_1d_array(booster) or isinstance(booster, list):
        importance = booster
    else:
        raise TypeError('booster must be Booster, LGBMModel or array instance.')

    if not len(importance):
        raise ValueError('Booster feature_importances are empty.')

    tuples = sorted(enumerate(importance), key=lambda x: x[1])
    if ignore_zero:
        tuples = [x for x in tuples if x[1] > 0]
    if max_num_features is not None and max_num_features > 0:
        tuples = tuples[-max_num_features:]
    labels, values = zip(*tuples)

    if ax is None:
        if figsize is not None and check_not_tuple_of_2_elements(figsize):
            raise TypeError('figsize must be a tuple of 2 elements.')
        _, ax = plt.subplots(1, 1, figsize=figsize)

    ylocs = np.arange(len(values))
    ax.barh(ylocs, values, align='center', height=height, **kwargs)

    for x, y in zip(values, ylocs):
        ax.text(x + 1, y, x, va='center')

    ax.set_yticks(ylocs)
    ax.set_yticklabels(labels)

    if xlim is not None:
        if check_not_tuple_of_2_elements(xlim):
            raise TypeError('xlim must be a tuple of 2 elements.')
    else:
        xlim = (0, max(values) * 1.1)
    ax.set_xlim(xlim)

    if ylim is not None:
        if check_not_tuple_of_2_elements(ylim):
            raise TypeError('ylim must be a tuple of 2 elements.')
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


def _to_graphviz(graph, tree_info, show_info):
    """Convert specified tree to graphviz instance."""

    def add(root, parent=None, decision=None):
        """recursively add node or edge"""
        if 'split_index' in root:  # non-leaf
            name = 'split' + str(root['split_index'])
            label = 'split_feature:' + str(root['split_feature'])
            label += '\nthreshold:' + str(root['threshold'])
            for info in show_info:
                if info in {'split_gain', 'internal_value', 'internal_count'}:
                    label += '\n' + info + ':' + str(root[info])
            graph.node(name, label=label)
            if root['decision_type'] == 'no_greater':
                l_dec, r_dec = '<=', '>'
            elif root['decision_type'] == 'is':
                l_dec, r_dec = 'is', "isn't"
            else:
                raise ValueError('Invalid decision type in tree model.')
            add(root['left_child'], name, l_dec)
            add(root['right_child'], name, r_dec)
        else:  # leaf
            name = 'left' + str(root['leaf_index'])
            label = 'leaf_value:' + str(root['leaf_value'])
            if 'leaf_count' in show_info:
                label += '\nleaf_count:' + str(root['leaf_count'])
            graph.node(name, label=label)
        if parent is not None:
            graph.edge(parent, name, decision)

    add(tree_info['tree_structure'])
    return graph


def plot_tree(booster, ax=None, tree_index=0, figsize=None,
              graph_attr=None, node_attr=None, edge_attr=None,
              show_info=None):
    """Plot specified tree.

    Parameters
    ----------
    booster : Booster, LGBMModel
        Booster or LGBMModel instance.
    ax : matplotlib Axes
        Target axes instance. If None, new figure and axes will be created.
    tree_index : int, default 0
        Specify tree index of target tree.
    figsize : tuple
        Figure size.
    graph_attr : dict
        Mapping of (attribute, value) pairs for the graph.
    node_attr : dict
        Mapping of (attribute, value) pairs set for all nodes.
    edge_attr : dict
        Mapping of (attribute, value) pairs set for all edges.
    show_info : list
        Information shows on nodes.
        options: 'split_gain', 'internal_value', 'internal_count' or 'leaf_count'.

    Returns
    -------
    ax : matplotlib Axes
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.image as image
    except ImportError:
        raise ImportError('You must install matplotlib to plot tree.')

    try:
        from graphviz import Digraph
    except ImportError:
        raise ImportError('You must install graphviz to plot tree.')

    if ax is None:
        if figsize is not None and check_not_tuple_of_2_elements(figsize):
            raise TypeError('xlim must be a tuple of 2 elements.')
        _, ax = plt.subplots(1, 1, figsize=figsize)

    if isinstance(booster, LGBMModel):
        booster = booster.booster_
    elif not isinstance(booster, Booster):
        raise TypeError('booster must be Booster or LGBMModel.')

    tree_infos = booster.dump_model()['tree_info']

    if tree_index < len(tree_infos):
        tree_info = tree_infos[tree_index]
    else:
        raise IndexError('tree_index is out of range.')

    graph = Digraph(graph_attr=graph_attr, node_attr=node_attr, edge_attr=edge_attr)

    if show_info is None:
        show_info = []
    ret = _to_graphviz(graph, tree_info, show_info)

    s = BytesIO()
    s.write(ret.pipe(format='png'))
    s.seek(0)
    img = image.imread(s)

    ax.imshow(img)
    ax.axis('off')
    return ax
