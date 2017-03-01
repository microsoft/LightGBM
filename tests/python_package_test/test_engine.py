# coding: utf-8
# pylint: skip-file
import copy
import math
import os
import unittest

import lightgbm as lgb
import numpy as np
from sklearn.datasets import (load_boston, load_breast_cancer, load_digits,
                              load_iris)
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, TimeSeriesSplit

try:
    import pandas as pd
    IS_PANDAS_INSTALLED = True
except ImportError:
    IS_PANDAS_INSTALLED = False

try:
    import cPickle as pickle
except ImportError:
    import pickle


def multi_logloss(y_true, y_pred):
    return np.mean([-math.log(y_pred[i][y]) for i, y in enumerate(y_true)])


class template(object):
    @staticmethod
    def test_template(params={'objective': 'regression', 'metric': 'l2'},
                      X_y=load_boston(True), feval=mean_squared_error,
                      num_round=150, init_model=None, custom_eval=None,
                      early_stopping_rounds=10,
                      return_data=False, return_model=False):
        params['verbose'], params['seed'] = -1, 42
        X_train, X_test, y_train, y_test = train_test_split(*X_y, test_size=0.1, random_state=42)
        lgb_train = lgb.Dataset(X_train, y_train, params=params)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train, params=params)
        if return_data:
            return lgb_train, lgb_eval
        evals_result = {}
        gbm = lgb.train(params, lgb_train,
                        num_boost_round=num_round,
                        valid_sets=lgb_eval,
                        valid_names='eval',
                        verbose_eval=False,
                        feval=custom_eval,
                        evals_result=evals_result,
                        early_stopping_rounds=early_stopping_rounds,
                        init_model=init_model)
        if return_model:
            return gbm
        else:
            return evals_result, feval(y_test, gbm.predict(X_test, gbm.best_iteration))


class TestEngine(unittest.TestCase):

    def test_binary(self):
        X_y = load_breast_cancer(True)
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss'
        }
        evals_result, ret = template.test_template(params, X_y, log_loss)
        self.assertLess(ret, 0.15)
        self.assertAlmostEqual(min(evals_result['eval']['binary_logloss']), ret, places=5)

    def test_regreesion(self):
        evals_result, ret = template.test_template()
        ret **= 0.5
        self.assertLess(ret, 4)
        self.assertAlmostEqual(min(evals_result['eval']['l2']), ret, places=5)

    def test_multiclass(self):
        X_y = load_digits(10, True)
        params = {
            'objective': 'multiclass',
            'metric': 'multi_logloss',
            'num_class': 10
        }
        evals_result, ret = template.test_template(params, X_y, multi_logloss)
        self.assertLess(ret, 0.2)
        self.assertAlmostEqual(min(evals_result['eval']['multi_logloss']), ret, places=5)

    def test_continue_train_and_other(self):
        params = {
            'objective': 'regression',
            'metric': 'l1'
        }
        model_name = 'model.txt'
        gbm = template.test_template(params, num_round=20, return_model=True, early_stopping_rounds=-1)
        gbm.save_model(model_name)
        evals_result, ret = template.test_template(params, feval=mean_absolute_error,
                                                   num_round=80, init_model=model_name,
                                                   custom_eval=(lambda p, d: ('mae', mean_absolute_error(p, d.get_label()), False)))
        self.assertLess(ret, 3)
        self.assertAlmostEqual(min(evals_result['eval']['l1']), ret, places=5)
        for l1, mae in zip(evals_result['eval']['l1'], evals_result['eval']['mae']):
            self.assertAlmostEqual(l1, mae, places=5)
        self.assertIn('tree_info', gbm.dump_model())
        self.assertIsInstance(gbm.feature_importance(), np.ndarray)
        os.remove(model_name)

    def test_continue_train_multiclass(self):
        X_y = load_iris(True)
        params = {
            'objective': 'multiclass',
            'metric': 'multi_logloss',
            'num_class': 3
        }
        gbm = template.test_template(params, X_y, num_round=20, return_model=True, early_stopping_rounds=-1)
        evals_result, ret = template.test_template(params, X_y, feval=multi_logloss,
                                                   num_round=80, init_model=gbm)
        self.assertLess(ret, 1.5)
        self.assertAlmostEqual(min(evals_result['eval']['multi_logloss']), ret, places=5)

    def test_cv(self):
        lgb_train, _ = template.test_template(return_data=True)
        lgb.cv({'verbose': -1}, lgb_train, num_boost_round=20, nfold=5, shuffle=False,
               metrics='l1', verbose_eval=False,
               callbacks=[lgb.reset_parameter(learning_rate=lambda i: 0.1 - 0.001 * i)])
        tss = TimeSeriesSplit(3)
        lgb.cv({'verbose': -1}, lgb_train, num_boost_round=20, data_splitter=tss, nfold=5,  # test if wrong nfold is ignored
               metrics='l2', verbose_eval=False)

    def test_feature_name(self):
        lgb_train, _ = template.test_template(return_data=True)
        feature_names = ['f' + str(i) for i in range(13)]
        gbm = lgb.train({'verbose': -1}, lgb_train, num_boost_round=10, feature_name=feature_names)
        self.assertListEqual(feature_names, gbm.feature_name())

    def test_save_load_copy_pickle(self):
        gbm = template.test_template(num_round=20, return_model=True)
        _, ret_origin = template.test_template(init_model=gbm)
        other_ret = []
        gbm.save_model('lgb.model')
        other_ret.append(template.test_template(init_model='lgb.model')[1])
        gbm_load = lgb.Booster(model_file='lgb.model')
        other_ret.append(template.test_template(init_model=gbm_load)[1])
        other_ret.append(template.test_template(init_model=copy.copy(gbm))[1])
        other_ret.append(template.test_template(init_model=copy.deepcopy(gbm))[1])
        with open('lgb.pkl', 'wb') as f:
            pickle.dump(gbm, f)
        with open('lgb.pkl', 'rb') as f:
            gbm_pickle = pickle.load(f)
        other_ret.append(template.test_template(init_model=gbm_pickle)[1])
        gbm_pickles = pickle.loads(pickle.dumps(gbm))
        other_ret.append(template.test_template(init_model=gbm_pickles)[1])
        for ret in other_ret:
            self.assertAlmostEqual(ret_origin, ret, places=5)

    @unittest.skipIf(not IS_PANDAS_INSTALLED, 'pandas not installed')
    def test_pandas_categorical(self):
        X = pd.DataFrame({"A": np.random.permutation(['a', 'b', 'c', 'd'] * 75),  # str
                          "B": np.random.permutation([1, 2, 3] * 100),  # int
                          "C": np.random.permutation([0.1, 0.2, -0.1, -0.1, 0.2] * 60),  # float
                          "D": np.random.permutation([True, False] * 150)})  # bool
        y = np.random.permutation([0, 1] * 150)
        X_test = pd.DataFrame({"A": np.random.permutation(['a', 'b', 'e'] * 20),
                               "B": np.random.permutation([1, 3] * 30),
                               "C": np.random.permutation([0.1, -0.1, 0.2, 0.2] * 15),
                               "D": np.random.permutation([True, False] * 30)})
        for col in ["A", "B", "C", "D"]:
            X[col] = X[col].astype('category')
            X_test[col] = X_test[col].astype('category')
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbose': -1
        }
        lgb_train = lgb.Dataset(X, y)
        gbm0 = lgb.train(params, lgb_train, num_boost_round=10, verbose_eval=False)
        pred0 = list(gbm0.predict(X_test))
        lgb_train = lgb.Dataset(X, y)
        gbm1 = lgb.train(params, lgb_train, num_boost_round=10, verbose_eval=False,
                         categorical_feature=[0])
        pred1 = list(gbm1.predict(X_test))
        lgb_train = lgb.Dataset(X, y)
        gbm2 = lgb.train(params, lgb_train, num_boost_round=10, verbose_eval=False,
                         categorical_feature=['A'])
        pred2 = list(gbm2.predict(X_test))
        lgb_train = lgb.Dataset(X, y)
        gbm3 = lgb.train(params, lgb_train, num_boost_round=10, verbose_eval=False,
                         categorical_feature=['A', 'B', 'C', 'D'])
        pred3 = list(gbm3.predict(X_test))
        lgb_train = lgb.Dataset(X, y)
        gbm3.save_model('categorical.model')
        gbm4 = lgb.Booster(model_file='categorical.model')
        pred4 = list(gbm4.predict(X_test))
        self.assertListEqual(pred0, pred1)
        self.assertListEqual(pred0, pred2)
        self.assertListEqual(pred0, pred3)
        self.assertListEqual(pred0, pred4)


print("----------------------------------------------------------------------")
print("running test_engine.py")
unittest.main()
