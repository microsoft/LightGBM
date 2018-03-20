# coding: utf-8
# pylint: disable = C0103
"""Plotting Library."""
from __future__ import absolute_import

import warnings
from copy import deepcopy
from io import BytesIO

import numpy as np

from .basic import Booster
from .sklearn import LGBMModel


def check_not_tuple_of_2_elements(obj, obj_name='obj'):
    """check object is not tuple or does not have 2 elements"""
    if not isinstance(obj, tuple) or len(obj) != 2:
        raise TypeError('%s must be a tuple of 2 elements.' % obj_name)


def plot_importance(booster, ax=None, height=0.2,
                    xlim=None, ylim=None, title='Feature importance',
                    xlabel='Feature importance', ylabel='Features',
                    importance_type='split', max_num_features=None,
                    ignore_zero=True, figsize=None, grid=True, **kwargs):
    """Plot model's feature importances.

    Parameters
    ----------
    booster : Booster or LGBMModel
        Booster or LGBMModel instance which feature importance should be plotted.
    ax : matplotlib.axes.Axes or None, optional (default=None)
        Target axes instance.
        If None, new figure and axes will be created.
    height : float, optional (default=0.2)
        Bar height, passed to ``ax.barh()``.
    xlim : tuple of 2 elements or None, optional (default=None)
        Tuple passed to ``ax.xlim()``.
    ylim : tuple of 2 elements or None, optional (default=None)
        Tuple passed to ``ax.ylim()``.
    title : string or None, optional (default="Feature importance")
        Axes title.
        If None, title is disabled.
    xlabel : string or None, optional (default="Feature importance")
        X-axis title label.
        If None, title is disabled.
    ylabel : string or None, optional (default="Features")
        Y-axis title label.
        If None, title is disabled.
    importance_type : string, optional (default="split")
        How the importance is calculated.
        If "split", result contains numbers of times the feature is used in a model.
        If "gain", result contains total gains of splits which use the feature.
    max_num_features : int or None, optional (default=None)
        Max number of top features displayed on plot.
        If None or <1, all features will be displayed.
    ignore_zero : bool, optional (default=True)
        Whether to ignore features with zero importance.
    figsize : tuple of 2 elements or None, optional (default=None)
        Figure size.
    grid : bool, optional (default=True)
        Whether to add a grid for axes.
    **kwargs : other parameters
        Other parameters passed to ``ax.barh()``.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The plot with model's feature importances.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError('You must install matplotlib to plot importance.')

    if isinstance(booster, LGBMModel):
        booster = booster.booster_
    elif not isinstance(booster, Booster):
        raise TypeError('booster must be Booster or LGBMModel.')

    importance = booster.feature_importance(importance_type=importance_type)
    feature_name = booster.feature_name()

    if not len(importance):
        raise ValueError('Booster feature_importances are empty.')

    tuples = sorted(zip(feature_name, importance), key=lambda x: x[1])
    if ignore_zero:
        tuples = [x for x in tuples if x[1] > 0]
    if max_num_features is not None and max_num_features > 0:
        tuples = tuples[-max_num_features:]
    labels, values = zip(*tuples)

    if ax is None:
        if figsize is not None:
            check_not_tuple_of_2_elements(figsize, 'figsize')
        _, ax = plt.subplots(1, 1, figsize=figsize)

    ylocs = np.arange(len(values))
    ax.barh(ylocs, values, align='center', height=height, **kwargs)

    for x, y in zip(values, ylocs):
        ax.text(x + 1, y, x, va='center')

    ax.set_yticks(ylocs)
    ax.set_yticklabels(labels)

    if xlim is not None:
        check_not_tuple_of_2_elements(xlim, 'xlim')
    else:
        xlim = (0, max(values) * 1.1)
    ax.set_xlim(xlim)

    if ylim is not None:
        check_not_tuple_of_2_elements(ylim, 'ylim')
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


def plot_metric(booster, metric=None, dataset_names=None,
                ax=None, xlim=None, ylim=None,
                title='Metric during training',
                xlabel='Iterations', ylabel='auto',
                figsize=None, grid=True):
    """Plot one metric during training.

    Parameters
    ----------
    booster : dict or LGBMModel
        Dictionary returned from ``lightgbm.train()`` or LGBMModel instance.
    metric : string or None, optional (default=None)
        The metric name to plot.
        Only one metric supported because different metrics have various scales.
        If None, first metric picked from dictionary (according to hashcode).
    dataset_names : list of strings or None, optional (default=None)
        List of the dataset names which are used to calculate metric to plot.
        If None, all datasets are used.
    ax : matplotlib.axes.Axes or None, optional (default=None)
        Target axes instance.
        If None, new figure and axes will be created.
    xlim : tuple of 2 elements or None, optional (default=None)
        Tuple passed to ``ax.xlim()``.
    ylim : tuple of 2 elements or None, optional (default=None)
        Tuple passed to ``ax.ylim()``.
    title : string or None, optional (default="Metric during training")
        Axes title.
        If None, title is disabled.
    xlabel : string or None, optional (default="Iterations")
        X-axis title label.
        If None, title is disabled.
    ylabel : string or None, optional (default="auto")
        Y-axis title label.
        If 'auto', metric name is used.
        If None, title is disabled.
    figsize : tuple of 2 elements or None, optional (default=None)
        Figure size.
    grid : bool, optional (default=True)
        Whether to add a grid for axes.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The plot with metric's history over the training.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError('You must install matplotlib to plot metric.')

    if isinstance(booster, LGBMModel):
        eval_results = deepcopy(booster.evals_result_)
    elif isinstance(booster, dict):
        eval_results = deepcopy(booster)
    else:
        raise TypeError('booster must be dict or LGBMModel.')

    num_data = len(eval_results)

    if not num_data:
        raise ValueError('eval results cannot be empty.')

    if ax is None:
        if figsize is not None:
            check_not_tuple_of_2_elements(figsize, 'figsize')
        _, ax = plt.subplots(1, 1, figsize=figsize)

    if dataset_names is None:
        dataset_names = iter(eval_results.keys())
    elif not isinstance(dataset_names, (list, tuple, set)) or not dataset_names:
        raise ValueError('dataset_names should be iterable and cannot be empty')
    else:
        dataset_names = iter(dataset_names)

    name = next(dataset_names)  # take one as sample
    metrics_for_one = eval_results[name]
    num_metric = len(metrics_for_one)
    if metric is None:
        if num_metric > 1:
            msg = """more than one metric available, picking one to plot."""
            warnings.warn(msg, stacklevel=2)
        metric, results = metrics_for_one.popitem()
    else:
        if metric not in metrics_for_one:
            raise KeyError('No given metric in eval results.')
        results = metrics_for_one[metric]
    num_iteration, max_result, min_result = len(results), max(results), min(results)
    x_ = range(num_iteration)
    ax.plot(x_, results, label=name)

    for name in dataset_names:
        metrics_for_one = eval_results[name]
        results = metrics_for_one[metric]
        max_result, min_result = max(max(results), max_result), min(min(results), min_result)
        ax.plot(x_, results, label=name)

    ax.legend(loc='best')

    if xlim is not None:
        check_not_tuple_of_2_elements(xlim, 'xlim')
    else:
        xlim = (0, num_iteration)
    ax.set_xlim(xlim)

    if ylim is not None:
        check_not_tuple_of_2_elements(ylim, 'ylim')
    else:
        range_result = max_result - min_result
        ylim = (min_result - range_result * 0.2, max_result + range_result * 0.2)
    ax.set_ylim(ylim)

    if ylabel == 'auto':
        ylabel = metric

    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.grid(grid)
    return ax


def _to_graphviz(tree_info, show_info, feature_names,
                 name=None, comment=None, filename=None, directory=None,
                 format=None, engine=None, encoding=None, graph_attr=None,
                 node_attr=None, edge_attr=None, body=None, strict=False):
    """Convert specified tree to graphviz instance.

    See:
      - http://graphviz.readthedocs.io/en/stable/api.html#digraph
    """
    try:
        from graphviz import Digraph
    except ImportError:
        raise ImportError('You must install graphviz to plot tree.')

    def add(root, parent=None, decision=None):
        """recursively add node or edge"""
        if 'split_index' in root:  # non-leaf
            name = 'split' + str(root['split_index'])
            if feature_names is not None:
                label = 'split_feature_name:' + str(feature_names[root['split_feature']])
            else:
                label = 'split_feature_index:' + str(root['split_feature'])
            label += r'\nthreshold:' + str(root['threshold'])
            for info in show_info:
                if info in {'split_gain', 'internal_value', 'internal_count'}:
                    label += r'\n' + info + ':' + str(root[info])
            graph.node(name, label=label)
            if root['decision_type'] == '<=':
                l_dec, r_dec = '<=', '>'
            elif root['decision_type'] == '==':
                l_dec, r_dec = 'is', "isn't"
            else:
                raise ValueError('Invalid decision type in tree model.')
            add(root['left_child'], name, l_dec)
            add(root['right_child'], name, r_dec)
        else:  # leaf
            name = 'leaf' + str(root['leaf_index'])
            label = 'leaf_index:' + str(root['leaf_index'])
            label += r'\nleaf_value:' + str(root['leaf_value'])
            if 'leaf_count' in show_info:
                label += r'\nleaf_count:' + str(root['leaf_count'])
            graph.node(name, label=label)
        if parent is not None:
            graph.edge(parent, name, decision)

    graph = Digraph(name=name, comment=comment, filename=filename, directory=directory,
                    format=format, engine=engine, encoding=encoding, graph_attr=graph_attr,
                    node_attr=node_attr, edge_attr=edge_attr, body=body, strict=strict)
    add(tree_info['tree_structure'])

    return graph


def create_tree_digraph(booster, tree_index=0, show_info=None,
                        name=None, comment=None, filename=None, directory=None,
                        format=None, engine=None, encoding=None, graph_attr=None,
                        node_attr=None, edge_attr=None, body=None, strict=False):
    """Create a digraph representation of specified tree.

    Note
    ----
    For more information please visit
    http://graphviz.readthedocs.io/en/stable/api.html#digraph.

    Parameters
    ----------
    booster : Booster or LGBMModel
        Booster or LGBMModel instance.
    tree_index : int, optional (default=0)
        The index of a target tree to convert.
    show_info : list or None, optional (default=None)
        What information should be showed on nodes.
        Possible values of list items: 'split_gain', 'internal_value', 'internal_count', 'leaf_count'.
    name : string or None, optional (default=None)
        Graph name used in the source code.
    comment : string or None, optional (default=None)
        Comment added to the first line of the source.
    filename : string or None, optional (default=None)
        Filename for saving the source.
        If None, ``name`` + '.gv' is used.
    directory : string or None, optional (default=None)
        (Sub)directory for source saving and rendering.
    format : string or None, optional (default=None)
        Rendering output format ('pdf', 'png', ...).
    engine : string or None, optional (default=None)
        Layout command used ('dot', 'neato', ...).
    encoding : string or None, optional (default=None)
        Encoding for saving the source.
    graph_attr : dict or None, optional (default=None)
        Mapping of (attribute, value) pairs set for the graph.
    node_attr : dict or None, optional (default=None)
        Mapping of (attribute, value) pairs set for all nodes.
    edge_attr : dict or None, optional (default=None)
        Mapping of (attribute, value) pairs set for all edges.
    body : list of strings or None, optional (default=None)
        Lines to add to the graph body.
    strict : bool, optional (default=False)
        Whether rendering should merge multi-edges.

    Returns
    -------
    graph : graphviz.Digraph
        The digraph representation of specified tree.
    """
    if isinstance(booster, LGBMModel):
        booster = booster.booster_
    elif not isinstance(booster, Booster):
        raise TypeError('booster must be Booster or LGBMModel.')

    model = booster.dump_model()
    tree_infos = model['tree_info']
    if 'feature_names' in model:
        feature_names = model['feature_names']
    else:
        feature_names = None

    if tree_index < len(tree_infos):
        tree_info = tree_infos[tree_index]
    else:
        raise IndexError('tree_index is out of range.')

    if show_info is None:
        show_info = []

    graph = _to_graphviz(tree_info, show_info, feature_names,
                         name=name, comment=comment, filename=filename, directory=directory,
                         format=format, engine=engine, encoding=encoding, graph_attr=graph_attr,
                         node_attr=node_attr, edge_attr=edge_attr, body=body, strict=strict)

    return graph


def plot_tree(booster, ax=None, tree_index=0, figsize=None,
              graph_attr=None, node_attr=None, edge_attr=None,
              show_info=None):
    """Plot specified tree.

    Parameters
    ----------
    booster : Booster or LGBMModel
        Booster or LGBMModel instance to be plotted.
    ax : matplotlib.axes.Axes or None, optional (default=None)
        Target axes instance.
        If None, new figure and axes will be created.
    tree_index : int, optional (default=0)
        The index of a target tree to plot.
    figsize : tuple of 2 elements or None, optional (default=None)
        Figure size.
    graph_attr : dict or None, optional (default=None)
        Mapping of (attribute, value) pairs set for the graph.
    node_attr : dict or None, optional (default=None)
        Mapping of (attribute, value) pairs set for all nodes.
    edge_attr : dict or None, optional (default=None)
        Mapping of (attribute, value) pairs set for all edges.
    show_info : list or None, optional (default=None)
        What information should be showed on nodes.
        Possible values of list items: 'split_gain', 'internal_value', 'internal_count', 'leaf_count'.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The plot with single tree.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.image as image
    except ImportError:
        raise ImportError('You must install matplotlib to plot tree.')

    if ax is None:
        if figsize is not None:
            check_not_tuple_of_2_elements(figsize, 'figsize')
        _, ax = plt.subplots(1, 1, figsize=figsize)

    graph = create_tree_digraph(
        booster=booster,
        tree_index=tree_index,
        graph_attr=graph_attr,
        node_attr=node_attr,
        edge_attr=edge_attr,
        show_info=show_info
    )

    s = BytesIO()
    s.write(graph.pipe(format='png'))
    s.seek(0)
    img = image.imread(s)

    ax.imshow(img)
    ax.axis('off')
    return ax
