# coding: utf-8
# pylint: skip-file
import unittest

import lightgbm as lgb
from lightgbm.compat import MATPLOTLIB_INSTALLED, GRAPHVIZ_INSTALLED
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

if MATPLOTLIB_INSTALLED:
    import matplotlib
    matplotlib.use('Agg')
if GRAPHVIZ_INSTALLED:
    import graphviz


class TestBasic(unittest.TestCase):

    def setUp(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(*load_breast_cancer(True),
                                                                                test_size=0.1, random_state=1)
        self.train_data = lgb.Dataset(self.X_train, self.y_train)
        self.params = {
            "objective": "binary",
            "verbose": -1,
            "num_leaves": 3
        }

    @unittest.skipIf(not MATPLOTLIB_INSTALLED, 'matplotlib is not installed')
    def test_plot_importance(self):
        gbm0 = lgb.train(self.params, self.train_data, num_boost_round=10)
        ax0 = lgb.plot_importance(gbm0)
        self.assertIsInstance(ax0, matplotlib.axes.Axes)
        self.assertEqual(ax0.get_title(), 'Feature importance')
        self.assertEqual(ax0.get_xlabel(), 'Feature importance')
        self.assertEqual(ax0.get_ylabel(), 'Features')
        self.assertLessEqual(len(ax0.patches), 30)

        gbm1 = lgb.LGBMClassifier(n_estimators=10, num_leaves=3, silent=True)
        gbm1.fit(self.X_train, self.y_train)

        ax1 = lgb.plot_importance(gbm1, color='r', title='t', xlabel='x', ylabel='y')
        self.assertIsInstance(ax1, matplotlib.axes.Axes)
        self.assertEqual(ax1.get_title(), 't')
        self.assertEqual(ax1.get_xlabel(), 'x')
        self.assertEqual(ax1.get_ylabel(), 'y')
        self.assertLessEqual(len(ax1.patches), 30)
        for patch in ax1.patches:
            self.assertTupleEqual(patch.get_facecolor(), (1., 0, 0, 1.))  # red

        ax2 = lgb.plot_importance(gbm0, color=['r', 'y', 'g', 'b'],
                                  title=None, xlabel=None, ylabel=None)
        self.assertIsInstance(ax2, matplotlib.axes.Axes)
        self.assertEqual(ax2.get_title(), '')
        self.assertEqual(ax2.get_xlabel(), '')
        self.assertEqual(ax2.get_ylabel(), '')
        self.assertLessEqual(len(ax2.patches), 30)
        self.assertTupleEqual(ax2.patches[0].get_facecolor(), (1., 0, 0, 1.))  # r
        self.assertTupleEqual(ax2.patches[1].get_facecolor(), (.75, .75, 0, 1.))  # y
        self.assertTupleEqual(ax2.patches[2].get_facecolor(), (0, .5, 0, 1.))  # g
        self.assertTupleEqual(ax2.patches[3].get_facecolor(), (0, 0, 1., 1.))  # b

    @unittest.skipIf(not MATPLOTLIB_INSTALLED, 'matplotlib is not installed')
    def test_plot_split_value_histogram(self):
        gbm0 = lgb.train(self.params, self.train_data, num_boost_round=10)
        ax0 = lgb.plot_split_value_histogram(gbm0, 27)
        self.assertIsInstance(ax0, matplotlib.axes.Axes)
        self.assertEqual(ax0.get_title(), 'Split value histogram for feature with index 27')
        self.assertEqual(ax0.get_xlabel(), 'Feature split value')
        self.assertEqual(ax0.get_ylabel(), 'Count')
        self.assertLessEqual(len(ax0.patches), 2)

        gbm1 = lgb.LGBMClassifier(n_estimators=10, num_leaves=3, silent=True)
        gbm1.fit(self.X_train, self.y_train)

        ax1 = lgb.plot_split_value_histogram(gbm1, gbm1.booster_.feature_name()[27], figsize=(10, 5),
                                             title='Histogram for feature @index/name@ @feature@',
                                             xlabel='x', ylabel='y', color='r')
        self.assertIsInstance(ax1, matplotlib.axes.Axes)
        self.assertEqual(ax1.get_title(),
                         'Histogram for feature name {}'.format(gbm1.booster_.feature_name()[27]))
        self.assertEqual(ax1.get_xlabel(), 'x')
        self.assertEqual(ax1.get_ylabel(), 'y')
        self.assertLessEqual(len(ax1.patches), 2)
        for patch in ax1.patches:
            self.assertTupleEqual(patch.get_facecolor(), (1., 0, 0, 1.))  # red

        ax2 = lgb.plot_split_value_histogram(gbm0, 27, bins=10, color=['r', 'y', 'g', 'b'],
                                             title=None, xlabel=None, ylabel=None)
        self.assertIsInstance(ax2, matplotlib.axes.Axes)
        self.assertEqual(ax2.get_title(), '')
        self.assertEqual(ax2.get_xlabel(), '')
        self.assertEqual(ax2.get_ylabel(), '')
        self.assertEqual(len(ax2.patches), 10)
        self.assertTupleEqual(ax2.patches[0].get_facecolor(), (1., 0, 0, 1.))  # r
        self.assertTupleEqual(ax2.patches[1].get_facecolor(), (.75, .75, 0, 1.))  # y
        self.assertTupleEqual(ax2.patches[2].get_facecolor(), (0, .5, 0, 1.))  # g
        self.assertTupleEqual(ax2.patches[3].get_facecolor(), (0, 0, 1., 1.))  # b

        self.assertRaises(ValueError, lgb.plot_split_value_histogram, gbm0, 0)  # was not used in splitting

    @unittest.skipIf(not MATPLOTLIB_INSTALLED or not GRAPHVIZ_INSTALLED, 'matplotlib or graphviz is not installed')
    def test_plot_tree(self):
        gbm = lgb.LGBMClassifier(n_estimators=10, num_leaves=3, silent=True)
        gbm.fit(self.X_train, self.y_train, verbose=False)

        self.assertRaises(IndexError, lgb.plot_tree, gbm, tree_index=83)

        ax = lgb.plot_tree(gbm, tree_index=3, figsize=(15, 8), show_info=['split_gain'])
        self.assertIsInstance(ax, matplotlib.axes.Axes)
        w, h = ax.axes.get_figure().get_size_inches()
        self.assertEqual(int(w), 15)
        self.assertEqual(int(h), 8)

    @unittest.skipIf(not GRAPHVIZ_INSTALLED, 'graphviz is not installed')
    def test_create_tree_digraph(self):
        constraints = [-1, 1] * int(self.X_train.shape[1] / 2)
        gbm = lgb.LGBMClassifier(n_estimators=10, num_leaves=3, silent=True, monotone_constraints=constraints)
        gbm.fit(self.X_train, self.y_train, verbose=False)

        self.assertRaises(IndexError, lgb.create_tree_digraph, gbm, tree_index=83)

        graph = lgb.create_tree_digraph(gbm, tree_index=3,
                                        show_info=['split_gain', 'internal_value', 'internal_weight'],
                                        name='Tree4', node_attr={'color': 'red'})
        graph.render(view=False)
        self.assertIsInstance(graph, graphviz.Digraph)
        self.assertEqual(graph.name, 'Tree4')
        self.assertEqual(graph.filename, 'Tree4.gv')
        self.assertEqual(len(graph.node_attr), 1)
        self.assertEqual(graph.node_attr['color'], 'red')
        self.assertEqual(len(graph.graph_attr), 0)
        self.assertEqual(len(graph.edge_attr), 0)
        graph_body = ''.join(graph.body)
        self.assertIn('leaf', graph_body)
        self.assertIn('gain', graph_body)
        self.assertIn('value', graph_body)
        self.assertIn('weight', graph_body)
        self.assertIn('#ffdddd', graph_body)
        self.assertIn('#ddffdd', graph_body)
        self.assertNotIn('data', graph_body)
        self.assertNotIn('count', graph_body)

    @unittest.skipIf(not MATPLOTLIB_INSTALLED, 'matplotlib is not installed')
    def test_plot_metrics(self):
        test_data = lgb.Dataset(self.X_test, self.y_test, reference=self.train_data)
        self.params.update({"metric": {"binary_logloss", "binary_error"}})

        evals_result0 = {}
        gbm0 = lgb.train(self.params, self.train_data,
                         valid_sets=[self.train_data, test_data],
                         valid_names=['v1', 'v2'],
                         num_boost_round=10,
                         evals_result=evals_result0,
                         verbose_eval=False)
        ax0 = lgb.plot_metric(evals_result0)
        self.assertIsInstance(ax0, matplotlib.axes.Axes)
        self.assertEqual(ax0.get_title(), 'Metric during training')
        self.assertEqual(ax0.get_xlabel(), 'Iterations')
        self.assertIn(ax0.get_ylabel(), {'binary_logloss', 'binary_error'})
        ax0 = lgb.plot_metric(evals_result0, metric='binary_error')
        ax0 = lgb.plot_metric(evals_result0, metric='binary_logloss', dataset_names=['v2'])

        evals_result1 = {}
        gbm1 = lgb.train(self.params, self.train_data,
                         num_boost_round=10,
                         evals_result=evals_result1,
                         verbose_eval=False)
        self.assertRaises(ValueError, lgb.plot_metric, evals_result1)

        gbm2 = lgb.LGBMClassifier(n_estimators=10, num_leaves=3, silent=True)
        gbm2.fit(self.X_train, self.y_train, eval_set=[(self.X_test, self.y_test)], verbose=False)
        ax2 = lgb.plot_metric(gbm2, title=None, xlabel=None, ylabel=None)
        self.assertIsInstance(ax2, matplotlib.axes.Axes)
        self.assertEqual(ax2.get_title(), '')
        self.assertEqual(ax2.get_xlabel(), '')
        self.assertEqual(ax2.get_ylabel(), '')
