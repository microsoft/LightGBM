from sklearn import datasets
import lightgbm as lgb
import numpy as np
import unittest


def top_k_error(y_true, y_pred, k):
    top_k = np.argpartition(-y_pred, k)[:, :k]
    num_correct = 0
    for i in range(len(y_true)):
        if y_true[i] in top_k[i]:
            num_correct += 1
    return 1 - num_correct / len(y_true)


class TestMultiMetrics(unittest.TestCase):

    def test_k_1(self):
        # test that default gives
        X, y = datasets.load_digits(return_X_y=True)
        params = {'objective': 'multiclass', 'num_classes': 10, 'metric': 'multi_error', 'num_leaves': 4, 'seed': 0}
        lgb_data = lgb.Dataset(X, label=y)
        results = {}
        est = lgb.train(params, lgb_data, valid_sets=[lgb_data], valid_names=['train'], evals_result=results)
        predict_default = est.predict(X)
        params = {'objective': 'multiclass', 'num_classes': 10, 'metric': 'multi_error', 'top_k_threshold': 1,
                  'num_leaves': 4, 'seed': 0}
        results = {}
        est = lgb.train(params, lgb_data, valid_sets=[lgb_data], valid_names=['train'], evals_result=results)
        predict_1 = est.predict(X)
        # check that default gives same result as k = 1
        self.assertTrue(np.max(np.abs(predict_1 - predict_default)) < 1e-5)
        # check against independent calculation
        err = top_k_error(y, predict_default, 1)
        self.assertTrue(abs(results['train']['multi_error'][-1] - err) < 1e-5)

    def test_k_2(self):
        X, y = datasets.load_digits(return_X_y=True)

        params = {'objective': 'multiclass', 'num_classes': 10, 'metric': 'multi_error', 'top_k_threshold': 2,
                  'num_leaves': 4, 'seed': 0}
        lgb_data = lgb.Dataset(X, label=y)
        results = {}
        est = lgb.train(params, lgb_data, valid_sets=[lgb_data], valid_names=['train'], evals_result=results)
        predict_2 = est.predict(X)
        err = top_k_error(y, predict_2, 2)
        self.assertTrue(abs(results['train']['multi_error'][-1] - err) < 1e-5)
