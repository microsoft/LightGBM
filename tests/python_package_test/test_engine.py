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
from sklearn.model_selection import train_test_split

try:
    import cPickle as pickle
except:
    import pickle


def multi_logloss(y_true, y_pred):
    return np.mean([-math.log(y_pred[i][y]) for i, y in enumerate(y_true)])


def test_template(params={'objective': 'regression', 'metric': 'l2'},
                  X_y=load_boston(True), feval=mean_squared_error,
                  num_round=100, init_model=None, custom_eval=None,
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
        evals_result, ret = test_template(params, X_y, log_loss)
        self.assertLess(ret, 0.15)
        self.assertAlmostEqual(min(evals_result['eval']['binary_logloss']), ret, places=5)

    def test_regreesion(self):
        evals_result, ret = test_template()
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
        evals_result, ret = test_template(params, X_y, multi_logloss)
        self.assertLess(ret, 0.2)
        self.assertAlmostEqual(min(evals_result['eval']['multi_logloss']), ret, places=5)

    def test_continue_train_and_other(self):
        params = {
            'objective': 'regression',
            'metric': 'l1'
        }
        model_name = 'model.txt'
        gbm = test_template(params, num_round=20, return_model=True, early_stopping_rounds=-1)
        gbm.save_model(model_name)
        evals_result, ret = test_template(params, feval=mean_absolute_error,
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
        gbm = test_template(params, X_y, num_round=20, return_model=True, early_stopping_rounds=-1)
        evals_result, ret = test_template(params, X_y, feval=multi_logloss,
                                          num_round=80, init_model=gbm)
        self.assertLess(ret, 1.5)
        self.assertAlmostEqual(min(evals_result['eval']['multi_logloss']), ret, places=5)

    def test_cv(self):
        lgb_train, _ = test_template(return_data=True)
        lgb.cv({'verbose': -1}, lgb_train, num_boost_round=20, nfold=5,
               metrics='l1', verbose_eval=False,
               callbacks=[lgb.reset_parameter(learning_rate=lambda i: 0.1 - 0.001 * i)])

    def test_save_load_copy_pickle(self):
        gbm = test_template(num_round=20, return_model=True)
        _, ret_origin = test_template(init_model=gbm)
        other_ret = []
        gbm.save_model('lgb.model')
        other_ret.append(test_template(init_model='lgb.model')[1])
        gbm_load = lgb.Booster(model_file='lgb.model')
        other_ret.append(test_template(init_model=gbm_load)[1])
        other_ret.append(test_template(init_model=copy.copy(gbm))[1])
        other_ret.append(test_template(init_model=copy.deepcopy(gbm))[1])
        with open('lgb.pkl', 'wb') as f:
            pickle.dump(gbm, f)
        with open('lgb.pkl', 'rb') as f:
            gbm_pickle = pickle.load(f)
        other_ret.append(test_template(init_model=gbm_pickle)[1])
        gbm_pickles = pickle.loads(pickle.dumps(gbm))
        other_ret.append(test_template(init_model=gbm_pickles)[1])
        for ret in other_ret:
            self.assertAlmostEqual(ret_origin, ret, places=5)


print("----------------------------------------------------------------------")
print("running test_engine.py")
unittest.main()
