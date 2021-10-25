# coding: utf-8
"""Plotting library."""
from copy import deepcopy
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .basic import Booster, _log_warning
from .compat import GRAPHVIZ_INSTALLED, MATPLOTLIB_INSTALLED
from .sklearn import LGBMModel


def _check_not_tuple_of_2_elements(obj: Any, obj_name: str = 'obj') -> None:
    """Check object is not tuple or does not have 2 elements."""
    if not isinstance(obj, tuple) or len(obj) != 2:
        raise TypeError(f"{obj_name} must be a tuple of 2 elements.")


def _float2str(value: float, precision: Optional[int] = None) -> str:
    return (f"{value:.{precision}f}"
            if precision is not None and not isinstance(value, str)
            else str(value))


def plot_importance(
    booster: Union[Booster, LGBMModel],
    ax=None,
    height: float = 0.2,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    title: Optional[str] = 'Feature importance',
    xlabel: Optional[str] = 'Feature importance',
    ylabel: Optional[str] = 'Features',
    importance_type: str = 'auto',
    max_num_features: Optional[int] = None,
    ignore_zero: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: Optional[int] = None,
    grid: bool = True,
    precision: Optional[int] = 3,
    **kwargs: Any
) -> Any:
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
    title : str or None, optional (default="Feature importance")
        Axes title.
        If None, title is disabled.
    xlabel : str or None, optional (default="Feature importance")
        X-axis title label.
        If None, title is disabled.
        @importance_type@ placeholder can be used, and it will be replaced with the value of ``importance_type`` parameter.
    ylabel : str or None, optional (default="Features")
        Y-axis title label.
        If None, title is disabled.
    importance_type : str, optional (default="auto")
        How the importance is calculated.
        If "auto", if ``booster`` parameter is LGBMModel, ``booster.importance_type`` attribute is used; "split" otherwise.
        If "split", result contains numbers of times the feature is used in a model.
        If "gain", result contains total gains of splits which use the feature.
    max_num_features : int or None, optional (default=None)
        Max number of top features displayed on plot.
        If None or <1, all features will be displayed.
    ignore_zero : bool, optional (default=True)
        Whether to ignore features with zero importance.
    figsize : tuple of 2 elements or None, optional (default=None)
        Figure size.
    dpi : int or None, optional (default=None)
        Resolution of the figure.
    grid : bool, optional (default=True)
        Whether to add a grid for axes.
    precision : int or None, optional (default=3)
        Used to restrict the display of floating point values to a certain precision.
    **kwargs
        Other parameters passed to ``ax.barh()``.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The plot with model's feature importances.
    """
    if MATPLOTLIB_INSTALLED:
        import matplotlib.pyplot as plt
    else:
        raise ImportError('You must install matplotlib and restart your session to plot importance.')

    if isinstance(booster, LGBMModel):
        if importance_type == "auto":
            importance_type = booster.importance_type
        booster = booster.booster_
    elif isinstance(booster, Booster):
        if importance_type == "auto":
            importance_type = "split"
    else:
        raise TypeError('booster must be Booster or LGBMModel.')

    importance = booster.feature_importance(importance_type=importance_type)
    feature_name = booster.feature_name()

    if not len(importance):
        raise ValueError("Booster's feature_importance is empty.")

    tuples = sorted(zip(feature_name, importance), key=lambda x: x[1])
    if ignore_zero:
        tuples = [x for x in tuples if x[1] > 0]
    if max_num_features is not None and max_num_features > 0:
        tuples = tuples[-max_num_features:]
    labels, values = zip(*tuples)

    if ax is None:
        if figsize is not None:
            _check_not_tuple_of_2_elements(figsize, 'figsize')
        _, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    ylocs = np.arange(len(values))
    ax.barh(ylocs, values, align='center', height=height, **kwargs)

    for x, y in zip(values, ylocs):
        ax.text(x + 1, y,
                _float2str(x, precision) if importance_type == 'gain' else x,
                va='center')

    ax.set_yticks(ylocs)
    ax.set_yticklabels(labels)

    if xlim is not None:
        _check_not_tuple_of_2_elements(xlim, 'xlim')
    else:
        xlim = (0, max(values) * 1.1)
    ax.set_xlim(xlim)

    if ylim is not None:
        _check_not_tuple_of_2_elements(ylim, 'ylim')
    else:
        ylim = (-1, len(values))
    ax.set_ylim(ylim)

    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        xlabel = xlabel.replace('@importance_type@', importance_type)
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.grid(grid)
    return ax


def plot_split_value_histogram(
    booster: Union[Booster, LGBMModel],
    feature: Union[int, str],
    bins: Union[int, str, None] = None,
    ax=None,
    width_coef: float = 0.8,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    title: Optional[str] = 'Split value histogram for feature with @index/name@ @feature@',
    xlabel: Optional[str] = 'Feature split value',
    ylabel: Optional[str] = 'Count',
    figsize: Optional[Tuple[float, float]] = None,
    dpi: Optional[int] = None,
    grid: bool = True,
    **kwargs: Any
) -> Any:
    """Plot split value histogram for the specified feature of the model.

    Parameters
    ----------
    booster : Booster or LGBMModel
        Booster or LGBMModel instance of which feature split value histogram should be plotted.
    feature : int or str
        The feature name or index the histogram is plotted for.
        If int, interpreted as index.
        If str, interpreted as name.
    bins : int, str or None, optional (default=None)
        The maximum number of bins.
        If None, the number of bins equals number of unique split values.
        If str, it should be one from the list of the supported values by ``numpy.histogram()`` function.
    ax : matplotlib.axes.Axes or None, optional (default=None)
        Target axes instance.
        If None, new figure and axes will be created.
    width_coef : float, optional (default=0.8)
        Coefficient for histogram bar width.
    xlim : tuple of 2 elements or None, optional (default=None)
        Tuple passed to ``ax.xlim()``.
    ylim : tuple of 2 elements or None, optional (default=None)
        Tuple passed to ``ax.ylim()``.
    title : str or None, optional (default="Split value histogram for feature with @index/name@ @feature@")
        Axes title.
        If None, title is disabled.
        @feature@ placeholder can be used, and it will be replaced with the value of ``feature`` parameter.
        @index/name@ placeholder can be used,
        and it will be replaced with ``index`` word in case of ``int`` type ``feature`` parameter
        or ``name`` word in case of ``str`` type ``feature`` parameter.
    xlabel : str or None, optional (default="Feature split value")
        X-axis title label.
        If None, title is disabled.
    ylabel : str or None, optional (default="Count")
        Y-axis title label.
        If None, title is disabled.
    figsize : tuple of 2 elements or None, optional (default=None)
        Figure size.
    dpi : int or None, optional (default=None)
        Resolution of the figure.
    grid : bool, optional (default=True)
        Whether to add a grid for axes.
    **kwargs
        Other parameters passed to ``ax.bar()``.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The plot with specified model's feature split value histogram.
    """
    if MATPLOTLIB_INSTALLED:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    else:
        raise ImportError('You must install matplotlib and restart your session to plot split value histogram.')

    if isinstance(booster, LGBMModel):
        booster = booster.booster_
    elif not isinstance(booster, Booster):
        raise TypeError('booster must be Booster or LGBMModel.')

    hist, bins = booster.get_split_value_histogram(feature=feature, bins=bins, xgboost_style=False)
    if np.count_nonzero(hist) == 0:
        raise ValueError('Cannot plot split value histogram, '
                         f'because feature {feature} was not used in splitting')
    width = width_coef * (bins[1] - bins[0])
    centred = (bins[:-1] + bins[1:]) / 2

    if ax is None:
        if figsize is not None:
            _check_not_tuple_of_2_elements(figsize, 'figsize')
        _, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    ax.bar(centred, hist, align='center', width=width, **kwargs)

    if xlim is not None:
        _check_not_tuple_of_2_elements(xlim, 'xlim')
    else:
        range_result = bins[-1] - bins[0]
        xlim = (bins[0] - range_result * 0.2, bins[-1] + range_result * 0.2)
    ax.set_xlim(xlim)

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    if ylim is not None:
        _check_not_tuple_of_2_elements(ylim, 'ylim')
    else:
        ylim = (0, max(hist) * 1.1)
    ax.set_ylim(ylim)

    if title is not None:
        title = title.replace('@feature@', str(feature))
        title = title.replace('@index/name@', ('name' if isinstance(feature, str) else 'index'))
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.grid(grid)
    return ax


def plot_metric(
    booster: Union[Dict, LGBMModel],
    metric: Optional[str] = None,
    dataset_names: Optional[List[str]] = None,
    ax=None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    title: Optional[str] = 'Metric during training',
    xlabel: Optional[str] = 'Iterations',
    ylabel: Optional[str] = '@metric@',
    figsize: Optional[Tuple[float, float]] = None,
    dpi: Optional[int] = None,
    grid: bool = True
) -> Any:
    """Plot one metric during training.

    Parameters
    ----------
    booster : dict or LGBMModel
        Dictionary returned from ``lightgbm.train()`` or LGBMModel instance.
    metric : str or None, optional (default=None)
        The metric name to plot.
        Only one metric supported because different metrics have various scales.
        If None, first metric picked from dictionary (according to hashcode).
    dataset_names : list of str, or None, optional (default=None)
        List of the dataset names which are used to calculate metric to plot.
        If None, all datasets are used.
    ax : matplotlib.axes.Axes or None, optional (default=None)
        Target axes instance.
        If None, new figure and axes will be created.
    xlim : tuple of 2 elements or None, optional (default=None)
        Tuple passed to ``ax.xlim()``.
    ylim : tuple of 2 elements or None, optional (default=None)
        Tuple passed to ``ax.ylim()``.
    title : str or None, optional (default="Metric during training")
        Axes title.
        If None, title is disabled.
    xlabel : str or None, optional (default="Iterations")
        X-axis title label.
        If None, title is disabled.
    ylabel : str or None, optional (default="@metric@")
        Y-axis title label.
        If 'auto', metric name is used.
        If None, title is disabled.
        @metric@ placeholder can be used, and it will be replaced with metric name.
    figsize : tuple of 2 elements or None, optional (default=None)
        Figure size.
    dpi : int or None, optional (default=None)
        Resolution of the figure.
    grid : bool, optional (default=True)
        Whether to add a grid for axes.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The plot with metric's history over the training.
    """
    if MATPLOTLIB_INSTALLED:
        import matplotlib.pyplot as plt
    else:
        raise ImportError('You must install matplotlib and restart your session to plot metric.')

    if isinstance(booster, LGBMModel):
        eval_results = deepcopy(booster.evals_result_)
    elif isinstance(booster, dict):
        eval_results = deepcopy(booster)
    elif isinstance(booster, Booster):
        raise TypeError("booster must be dict or LGBMModel. To use plot_metric with Booster type, first record the metrics using record_evaluation callback then pass that to plot_metric as argument `booster`")
    else:
        raise TypeError('booster must be dict or LGBMModel.')

    num_data = len(eval_results)

    if not num_data:
        raise ValueError('eval results cannot be empty.')

    if ax is None:
        if figsize is not None:
            _check_not_tuple_of_2_elements(figsize, 'figsize')
        _, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

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
            _log_warning("More than one metric available, picking one to plot.")
        metric, results = metrics_for_one.popitem()
    else:
        if metric not in metrics_for_one:
            raise KeyError('No given metric in eval results.')
        results = metrics_for_one[metric]
    num_iteration = len(results)
    max_result = max(results)
    min_result = min(results)
    x_ = range(num_iteration)
    ax.plot(x_, results, label=name)

    for name in dataset_names:
        metrics_for_one = eval_results[name]
        results = metrics_for_one[metric]
        max_result = max(max(results), max_result)
        min_result = min(min(results), min_result)
        ax.plot(x_, results, label=name)

    ax.legend(loc='best')

    if xlim is not None:
        _check_not_tuple_of_2_elements(xlim, 'xlim')
    else:
        xlim = (0, num_iteration)
    ax.set_xlim(xlim)

    if ylim is not None:
        _check_not_tuple_of_2_elements(ylim, 'ylim')
    else:
        range_result = max_result - min_result
        ylim = (min_result - range_result * 0.2, max_result + range_result * 0.2)
    ax.set_ylim(ylim)

    if ylabel == 'auto':
        _log_warning("'auto' value of 'ylabel' argument is deprecated and will be removed in a future release of LightGBM. "
                     "Use '@metric@' placeholder instead.")
        ylabel = '@metric@'

    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ylabel = ylabel.replace('@metric@', metric)
        ax.set_ylabel(ylabel)
    ax.grid(grid)
    return ax


def _to_graphviz(
    tree_info: Dict[str, Any],
    show_info: List[str],
    feature_names: Union[List[str], None],
    precision: Optional[int] = 3,
    orientation: str = 'horizontal',
    constraints: Optional[List[int]] = None,
    **kwargs: Any
) -> Any:
    """Convert specified tree to graphviz instance.

    See:
      - https://graphviz.readthedocs.io/en/stable/api.html#digraph
    """
    if GRAPHVIZ_INSTALLED:
        from graphviz import Digraph
    else:
        raise ImportError('You must install graphviz and restart your session to plot tree.')

    def add(root, total_count, parent=None, decision=None):
        """Recursively add node or edge."""
        if 'split_index' in root:  # non-leaf
            l_dec = 'yes'
            r_dec = 'no'
            if root['decision_type'] == '<=':
                lte_symbol = "&#8804;"
                operator = lte_symbol
            elif root['decision_type'] == '==':
                operator = "="
            else:
                raise ValueError('Invalid decision type in tree model.')
            name = f"split{root['split_index']}"
            if feature_names is not None:
                label = f"<B>{feature_names[root['split_feature']]}</B> {operator}"
            else:
                label = f"feature <B>{root['split_feature']}</B> {operator} "
            label += f"<B>{_float2str(root['threshold'], precision)}</B>"
            for info in ['split_gain', 'internal_value', 'internal_weight', "internal_count", "data_percentage"]:
                if info in show_info:
                    output = info.split('_')[-1]
                    if info in {'split_gain', 'internal_value', 'internal_weight'}:
                        label += f"<br/>{_float2str(root[info], precision)} {output}"
                    elif info == 'internal_count':
                        label += f"<br/>{output}: {root[info]}"
                    elif info == "data_percentage":
                        label += f"<br/>{_float2str(root['internal_count'] / total_count * 100, 2)}% of data"

            fillcolor = "white"
            style = ""
            if constraints:
                if constraints[root['split_feature']] == 1:
                    fillcolor = "#ddffdd"  # light green
                if constraints[root['split_feature']] == -1:
                    fillcolor = "#ffdddd"  # light red
                style = "filled"
            label = f"<{label}>"
            graph.node(name, label=label, shape="rectangle", style=style, fillcolor=fillcolor)
            add(root['left_child'], total_count, name, l_dec)
            add(root['right_child'], total_count, name, r_dec)
        else:  # leaf
            name = f"leaf{root['leaf_index']}"
            label = f"leaf {root['leaf_index']}: "
            label += f"<B>{_float2str(root['leaf_value'], precision)}</B>"
            if 'leaf_weight' in show_info:
                label += f"<br/>{_float2str(root['leaf_weight'], precision)} weight"
            if 'leaf_count' in show_info:
                label += f"<br/>count: {root['leaf_count']}"
            if "data_percentage" in show_info:
                label += f"<br/>{_float2str(root['leaf_count'] / total_count * 100, 2)}% of data"
            label = f"<{label}>"
            graph.node(name, label=label)
        if parent is not None:
            graph.edge(parent, name, decision)

    graph = Digraph(**kwargs)
    rankdir = "LR" if orientation == "horizontal" else "TB"
    graph.attr("graph", nodesep="0.05", ranksep="0.3", rankdir=rankdir)
    if "internal_count" in tree_info['tree_structure']:
        add(tree_info['tree_structure'], tree_info['tree_structure']["internal_count"])
    else:
        raise Exception("Cannot plot trees with no split")

    if constraints:
        # "#ddffdd" is light green, "#ffdddd" is light red
        legend = """<
            <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
             <TR>
              <TD COLSPAN="2"><B>Monotone constraints</B></TD>
             </TR>
             <TR>
              <TD>Increasing</TD>
              <TD BGCOLOR="#ddffdd"></TD>
             </TR>
             <TR>
              <TD>Decreasing</TD>
              <TD BGCOLOR="#ffdddd"></TD>
             </TR>
            </TABLE>
           >"""
        graph.node("legend", label=legend, shape="rectangle", color="white")
    return graph


def create_tree_digraph(
    booster: Union[Booster, LGBMModel],
    tree_index: int = 0,
    show_info: Optional[List[str]] = None,
    precision: Optional[int] = 3,
    orientation: str = 'horizontal',
    **kwargs: Any
) -> Any:
    """Create a digraph representation of specified tree.

    Each node in the graph represents a node in the tree.

    Non-leaf nodes have labels like ``Column_10 <= 875.9``, which means
    "this node splits on the feature named "Column_10", with threshold 875.9".

    Leaf nodes have labels like ``leaf 2: 0.422``, which means "this node is a
    leaf node, and the predicted value for records that fall into this node
    is 0.422". The number (``2``) is an internal unique identifier and doesn't
    have any special meaning.

    .. note::

        For more information please visit
        https://graphviz.readthedocs.io/en/stable/api.html#digraph.

    Parameters
    ----------
    booster : Booster or LGBMModel
        Booster or LGBMModel instance to be converted.
    tree_index : int, optional (default=0)
        The index of a target tree to convert.
    show_info : list of str, or None, optional (default=None)
        What information should be shown in nodes.

            - ``'split_gain'`` : gain from adding this split to the model
            - ``'internal_value'`` : raw predicted value that would be produced by this node if it was a leaf node
            - ``'internal_count'`` : number of records from the training data that fall into this non-leaf node
            - ``'internal_weight'`` : total weight of all nodes that fall into this non-leaf node
            - ``'leaf_count'`` : number of records from the training data that fall into this leaf node
            - ``'leaf_weight'`` : total weight (sum of hessian) of all observations that fall into this leaf node
            - ``'data_percentage'`` : percentage of training data that fall into this node
    precision : int or None, optional (default=3)
        Used to restrict the display of floating point values to a certain precision.
    orientation : str, optional (default='horizontal')
        Orientation of the tree.
        Can be 'horizontal' or 'vertical'.
    **kwargs
        Other parameters passed to ``Digraph`` constructor.
        Check https://graphviz.readthedocs.io/en/stable/api.html#digraph for the full list of supported parameters.

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

    monotone_constraints = model.get('monotone_constraints', None)

    if tree_index < len(tree_infos):
        tree_info = tree_infos[tree_index]
    else:
        raise IndexError('tree_index is out of range.')

    if show_info is None:
        show_info = []

    graph = _to_graphviz(tree_info, show_info, feature_names, precision,
                         orientation, monotone_constraints, **kwargs)

    return graph


def plot_tree(
    booster: Union[Booster, LGBMModel],
    ax=None,
    tree_index: int = 0,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: Optional[int] = None,
    show_info: Optional[List[str]] = None,
    precision: Optional[int] = 3,
    orientation: str = 'horizontal',
    **kwargs: Any
) -> Any:
    """Plot specified tree.

    Each node in the graph represents a node in the tree.

    Non-leaf nodes have labels like ``Column_10 <= 875.9``, which means
    "this node splits on the feature named "Column_10", with threshold 875.9".

    Leaf nodes have labels like ``leaf 2: 0.422``, which means "this node is a
    leaf node, and the predicted value for records that fall into this node
    is 0.422". The number (``2``) is an internal unique identifier and doesn't
    have any special meaning.

    .. note::

        It is preferable to use ``create_tree_digraph()`` because of its lossless quality
        and returned objects can be also rendered and displayed directly inside a Jupyter notebook.

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
    dpi : int or None, optional (default=None)
        Resolution of the figure.
    show_info : list of str, or None, optional (default=None)
        What information should be shown in nodes.

            - ``'split_gain'`` : gain from adding this split to the model
            - ``'internal_value'`` : raw predicted value that would be produced by this node if it was a leaf node
            - ``'internal_count'`` : number of records from the training data that fall into this non-leaf node
            - ``'internal_weight'`` : total weight of all nodes that fall into this non-leaf node
            - ``'leaf_count'`` : number of records from the training data that fall into this leaf node
            - ``'leaf_weight'`` : total weight (sum of hessian) of all observations that fall into this leaf node
            - ``'data_percentage'`` : percentage of training data that fall into this node
    precision : int or None, optional (default=3)
        Used to restrict the display of floating point values to a certain precision.
    orientation : str, optional (default='horizontal')
        Orientation of the tree.
        Can be 'horizontal' or 'vertical'.
    **kwargs
        Other parameters passed to ``Digraph`` constructor.
        Check https://graphviz.readthedocs.io/en/stable/api.html#digraph for the full list of supported parameters.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The plot with single tree.
    """
    if MATPLOTLIB_INSTALLED:
        import matplotlib.image as image
        import matplotlib.pyplot as plt
    else:
        raise ImportError('You must install matplotlib and restart your session to plot tree.')

    if ax is None:
        if figsize is not None:
            _check_not_tuple_of_2_elements(figsize, 'figsize')
        _, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    graph = create_tree_digraph(booster=booster, tree_index=tree_index,
                                show_info=show_info, precision=precision,
                                orientation=orientation, **kwargs)

    s = BytesIO()
    s.write(graph.pipe(format='png'))
    s.seek(0)
    img = image.imread(s)

    ax.imshow(img)
    ax.axis('off')
    return ax
