# coding: utf-8
# pylint: skip-file
import unittest

import lightgbm as lgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

try:
    import matplotlib
    matplotlib.use('Agg')
    matplotlib_installed = True
except ImportError:
    matplotlib_installed = False


class TestBasic(unittest.TestCase):

    @unittest.skipIf(not matplotlib_installed, 'matplotlib not installed')
    def test_plot_importance(self):
        X_train, _, y_train, _ = train_test_split(*load_breast_cancer(True), test_size=0.1, random_state=1)
        train_data = lgb.Dataset(X_train, y_train)

        params = {
            "objective": "binary",
            "verbose": -1,
            "num_leaves": 3
        }
        gbm0 = lgb.train(params, train_data, num_boost_round=10)
        ax0 = lgb.plot_importance(gbm0)
        self.assertIsInstance(ax0, matplotlib.axes.Axes)
        self.assertEqual(ax0.get_title(), 'Feature importance')
        self.assertEqual(ax0.get_xlabel(), 'Feature importance')
        self.assertEqual(ax0.get_ylabel(), 'Features')
        self.assertLessEqual(len(ax0.patches), 30)

        gbm1 = lgb.LGBMClassifier(n_estimators=10, num_leaves=3, silent=True)
        gbm1.fit(X_train, y_train)

        ax1 = lgb.plot_importance(gbm1, color='r', title='t', xlabel='x', ylabel='y')
        self.assertIsInstance(ax1, matplotlib.axes.Axes)
        self.assertEqual(ax1.get_title(), 't')
        self.assertEqual(ax1.get_xlabel(), 'x')
        self.assertEqual(ax1.get_ylabel(), 'y')
        self.assertLessEqual(len(ax1.patches), 30)
        for patch in ax1.patches:
            self.assertTupleEqual(patch.get_facecolor(), (1., 0, 0, 1.))  # red

        ax2 = lgb.plot_importance(gbm0.feature_importance(),
                                  color=['r', 'y', 'g', 'b'],
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

    @unittest.skip('Graphviz are not executables on Travis')
    def test_plot_tree(self):
        pass


print("----------------------------------------------------------------------")
print("running test_plotting.py")
unittest.main()
