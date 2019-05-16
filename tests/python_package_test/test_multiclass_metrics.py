from sklearn import datasets
import lightgbm as lgb
import numpy as np
import unittest


def top_k_error(y_true, y_pred, k):
    top_k = np.argpartition(-y_pred, k)[:, :k]
    num_correct = 0.0
    for i in range(len(y_true)):
        if y_true[i] in top_k[i]:
            num_correct += 1
    return 1 - num_correct / len(y_true)


class TestMultiMetrics(unittest.TestCase):

    def test_multi_class_error(self):
        X, y = datasets.load_digits(return_X_y=True)
        params = {'objective': 'multiclass', 'num_classes': 10, 'metric': 'multi_error', 'num_leaves': 4, 'seed': 0,
                  'num_rounds': 30, 'verbose': -1}
        lgb_data = lgb.Dataset(X, label=y)
        results = {}
        est = lgb.train(params, lgb_data, valid_sets=[lgb_data], valid_names=['train'], evals_result=results)
        predict_default = est.predict(X)
        params = {'objective': 'multiclass', 'num_classes': 10, 'metric': 'multi_error', 'top_k_threshold': 1,
                  'num_leaves': 4, 'seed': 0, 'num_rounds': 30, 'verbose': -1, 'metric_freq': 10}
        results = {}
        est = lgb.train(params, lgb_data, valid_sets=[lgb_data], valid_names=['train'], evals_result=results)
        predict_1 = est.predict(X)
        # check that default gives same result as k = 1
        np.testing.assert_array_almost_equal(predict_1, predict_default, 5)
        # check against independent calculation for k = 1
        err = top_k_error(y, predict_1, 1)
        np.testing.assert_almost_equal(results['train']['multi_error'][-1], err, 5)
        # check against independent calculation for k = 2
        params = {'objective': 'multiclass', 'num_classes': 10, 'metric': 'multi_error', 'top_k_threshold': 2,
                  'num_leaves': 4, 'seed': 0, 'num_rounds': 30, 'verbose': -1, 'metric_freq': 10}
        results = {}
        est = lgb.train(params, lgb_data, valid_sets=[lgb_data], valid_names=['train'], evals_result=results)
        predict_2 = est.predict(X)
        err = top_k_error(y, predict_2, 2)
        np.testing.assert_almost_equal(results['train']['multi_error'][-1], err, 5)
