# coding: utf-8
import pytest
from sklearn.model_selection import train_test_split

import lightgbm as lgb
from lightgbm.compat import GRAPHVIZ_INSTALLED, MATPLOTLIB_INSTALLED

if MATPLOTLIB_INSTALLED:
    import matplotlib
    matplotlib.use('Agg')
if GRAPHVIZ_INSTALLED:
    import graphviz

from .utils import load_breast_cancer


@pytest.fixture(scope="module")
def breast_cancer_split():
    return train_test_split(*load_breast_cancer(return_X_y=True),
                            test_size=0.1, random_state=1)


@pytest.fixture(scope="module")
def train_data(breast_cancer_split):
    X_train, _, y_train, _ = breast_cancer_split
    return lgb.Dataset(X_train, y_train)


@pytest.fixture
def params():
    return {"objective": "binary",
            "verbose": -1,
            "num_leaves": 3}


@pytest.mark.skipif(not MATPLOTLIB_INSTALLED, reason='matplotlib is not installed')
def test_plot_importance(params, breast_cancer_split, train_data):
    X_train, _, y_train, _ = breast_cancer_split

    gbm0 = lgb.train(params, train_data, num_boost_round=10)
    ax0 = lgb.plot_importance(gbm0)
    assert isinstance(ax0, matplotlib.axes.Axes)
    assert ax0.get_title() == 'Feature importance'
    assert ax0.get_xlabel() == 'Feature importance'
    assert ax0.get_ylabel() == 'Features'
    assert len(ax0.patches) <= 30

    gbm1 = lgb.LGBMClassifier(n_estimators=10, num_leaves=3, silent=True)
    gbm1.fit(X_train, y_train)

    ax1 = lgb.plot_importance(gbm1, color='r', title='t', xlabel='x', ylabel='y')
    assert isinstance(ax1, matplotlib.axes.Axes)
    assert ax1.get_title() == 't'
    assert ax1.get_xlabel() == 'x'
    assert ax1.get_ylabel() == 'y'
    assert len(ax1.patches) <= 30
    for patch in ax1.patches:
        assert patch.get_facecolor() == (1., 0, 0, 1.)  # red

    ax2 = lgb.plot_importance(gbm0, color=['r', 'y', 'g', 'b'],
                              title=None, xlabel=None, ylabel=None)
    assert isinstance(ax2, matplotlib.axes.Axes)
    assert ax2.get_title() == ''
    assert ax2.get_xlabel() == ''
    assert ax2.get_ylabel() == ''
    assert len(ax2.patches) <= 30
    assert ax2.patches[0].get_facecolor() == (1., 0, 0, 1.)  # r
    assert ax2.patches[1].get_facecolor() == (.75, .75, 0, 1.)  # y
    assert ax2.patches[2].get_facecolor() == (0, .5, 0, 1.)  # g
    assert ax2.patches[3].get_facecolor() == (0, 0, 1., 1.)  # b


@pytest.mark.skipif(not MATPLOTLIB_INSTALLED, reason='matplotlib is not installed')
def test_plot_split_value_histogram(params, breast_cancer_split, train_data):
    X_train, _, y_train, _ = breast_cancer_split

    gbm0 = lgb.train(params, train_data, num_boost_round=10)
    ax0 = lgb.plot_split_value_histogram(gbm0, 27)
    assert isinstance(ax0, matplotlib.axes.Axes)
    assert ax0.get_title() == 'Split value histogram for feature with index 27'
    assert ax0.get_xlabel() == 'Feature split value'
    assert ax0.get_ylabel() == 'Count'
    assert len(ax0.patches) <= 2

    gbm1 = lgb.LGBMClassifier(n_estimators=10, num_leaves=3, silent=True)
    gbm1.fit(X_train, y_train)

    ax1 = lgb.plot_split_value_histogram(gbm1, gbm1.booster_.feature_name()[27], figsize=(10, 5),
                                         title='Histogram for feature @index/name@ @feature@',
                                         xlabel='x', ylabel='y', color='r')
    assert isinstance(ax1, matplotlib.axes.Axes)
    title = f'Histogram for feature name {gbm1.booster_.feature_name()[27]}'
    assert ax1.get_title() == title
    assert ax1.get_xlabel() == 'x'
    assert ax1.get_ylabel() == 'y'
    assert len(ax1.patches) <= 2
    for patch in ax1.patches:
        assert patch.get_facecolor() == (1., 0, 0, 1.)  # red

    ax2 = lgb.plot_split_value_histogram(gbm0, 27, bins=10, color=['r', 'y', 'g', 'b'],
                                         title=None, xlabel=None, ylabel=None)
    assert isinstance(ax2, matplotlib.axes.Axes)
    assert ax2.get_title() == ''
    assert ax2.get_xlabel() == ''
    assert ax2.get_ylabel() == ''
    assert len(ax2.patches) == 10
    assert ax2.patches[0].get_facecolor() == (1., 0, 0, 1.)  # r
    assert ax2.patches[1].get_facecolor() == (.75, .75, 0, 1.)  # y
    assert ax2.patches[2].get_facecolor() == (0, .5, 0, 1.)  # g
    assert ax2.patches[3].get_facecolor() == (0, 0, 1., 1.)  # b

    with pytest.raises(ValueError):
        lgb.plot_split_value_histogram(gbm0, 0)  # was not used in splitting


@pytest.mark.skipif(not MATPLOTLIB_INSTALLED or not GRAPHVIZ_INSTALLED,
                    reason='matplotlib or graphviz is not installed')
def test_plot_tree(breast_cancer_split):
    X_train, _, y_train, _ = breast_cancer_split
    gbm = lgb.LGBMClassifier(n_estimators=10, num_leaves=3, silent=True)
    gbm.fit(X_train, y_train, verbose=False)

    with pytest.raises(IndexError):
        lgb.plot_tree(gbm, tree_index=83)

    ax = lgb.plot_tree(gbm, tree_index=3, figsize=(15, 8), show_info=['split_gain'])
    assert isinstance(ax, matplotlib.axes.Axes)
    w, h = ax.axes.get_figure().get_size_inches()
    assert int(w) == 15
    assert int(h) == 8


@pytest.mark.skipif(not GRAPHVIZ_INSTALLED, reason='graphviz is not installed')
def test_create_tree_digraph(breast_cancer_split):
    X_train, _, y_train, _ = breast_cancer_split

    constraints = [-1, 1] * int(X_train.shape[1] / 2)
    gbm = lgb.LGBMClassifier(n_estimators=10, num_leaves=3, silent=True, monotone_constraints=constraints)
    gbm.fit(X_train, y_train, verbose=False)

    with pytest.raises(IndexError):
        lgb.create_tree_digraph(gbm, tree_index=83)

    graph = lgb.create_tree_digraph(gbm, tree_index=3,
                                    show_info=['split_gain', 'internal_value', 'internal_weight'],
                                    name='Tree4', node_attr={'color': 'red'})
    graph.render(view=False)
    assert isinstance(graph, graphviz.Digraph)
    assert graph.name == 'Tree4'
    assert graph.filename == 'Tree4.gv'
    assert len(graph.node_attr) == 1
    assert graph.node_attr['color'] == 'red'
    assert len(graph.graph_attr) == 0
    assert len(graph.edge_attr) == 0
    graph_body = ''.join(graph.body)
    assert 'leaf' in graph_body
    assert 'gain' in graph_body
    assert 'value' in graph_body
    assert 'weight' in graph_body
    assert '#ffdddd' in graph_body
    assert '#ddffdd' in graph_body
    assert 'data' not in graph_body
    assert 'count' not in graph_body


@pytest.mark.skipif(not MATPLOTLIB_INSTALLED, reason='matplotlib is not installed')
def test_plot_metrics(params, breast_cancer_split, train_data):
    X_train, X_test, y_train, y_test = breast_cancer_split
    test_data = lgb.Dataset(X_test, y_test, reference=train_data)
    params.update({"metric": {"binary_logloss", "binary_error"}})

    evals_result0 = {}
    lgb.train(params, train_data,
              valid_sets=[train_data, test_data],
              valid_names=['v1', 'v2'],
              num_boost_round=10,
              evals_result=evals_result0,
              verbose_eval=False)
    ax0 = lgb.plot_metric(evals_result0)
    assert isinstance(ax0, matplotlib.axes.Axes)
    assert ax0.get_title() == 'Metric during training'
    assert ax0.get_xlabel() == 'Iterations'
    assert ax0.get_ylabel() in {'binary_logloss', 'binary_error'}
    ax0 = lgb.plot_metric(evals_result0, metric='binary_error')
    ax0 = lgb.plot_metric(evals_result0, metric='binary_logloss', dataset_names=['v2'])

    evals_result1 = {}
    lgb.train(params, train_data,
              num_boost_round=10,
              evals_result=evals_result1,
              verbose_eval=False)
    with pytest.raises(ValueError):
        lgb.plot_metric(evals_result1)

    gbm2 = lgb.LGBMClassifier(n_estimators=10, num_leaves=3, silent=True)
    gbm2.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    ax2 = lgb.plot_metric(gbm2, title=None, xlabel=None, ylabel=None)
    assert isinstance(ax2, matplotlib.axes.Axes)
    assert ax2.get_title() == ''
    assert ax2.get_xlabel() == ''
    assert ax2.get_ylabel() == ''
