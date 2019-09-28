# coding: utf-8
# pylint: skip-file
import copy
import itertools
import math
import os
import psutil
import random
import unittest

import lightgbm as lgb
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.datasets import (load_boston, load_breast_cancer, load_digits,
                              load_iris, load_svmlight_file)
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GroupKFold

try:
    import cPickle as pickle
except ImportError:
    import pickle


def multi_logloss(y_true, y_pred):
    return np.mean([-math.log(y_pred[i][y]) for i, y in enumerate(y_true)])


def top_k_error(y_true, y_pred, k):
    if k == y_pred.shape[1]:
        return 0
    max_rest = np.max(-np.partition(-y_pred, k)[:, k:], axis=1)
    return 1 - np.mean((y_pred[np.arange(len(y_true)), y_true] > max_rest))


class TestEngine(unittest.TestCase):
    def test_binary(self):
        X, y = load_breast_cancer(True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbose': -1,
            'num_iteration': 50  # test num_iteration in dict here
        }
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
        evals_result = {}
        gbm = lgb.train(params, lgb_train,
                        num_boost_round=20,
                        valid_sets=lgb_eval,
                        verbose_eval=False,
                        evals_result=evals_result)
        ret = log_loss(y_test, gbm.predict(X_test))
        self.assertLess(ret, 0.15)
        self.assertEqual(len(evals_result['valid_0']['binary_logloss']), 50)
        self.assertAlmostEqual(evals_result['valid_0']['binary_logloss'][-1], ret, places=5)

    def test_rf(self):
        X, y = load_breast_cancer(True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        params = {
            'boosting_type': 'rf',
            'objective': 'binary',
            'bagging_freq': 1,
            'bagging_fraction': 0.5,
            'feature_fraction': 0.5,
            'num_leaves': 50,
            'metric': 'binary_logloss',
            'verbose': -1
        }
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
        evals_result = {}
        gbm = lgb.train(params, lgb_train,
                        num_boost_round=50,
                        valid_sets=lgb_eval,
                        verbose_eval=False,
                        evals_result=evals_result)
        ret = log_loss(y_test, gbm.predict(X_test))
        self.assertLess(ret, 0.25)
        self.assertAlmostEqual(evals_result['valid_0']['binary_logloss'][-1], ret, places=5)

    def test_regression(self):
        X, y = load_boston(True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        params = {
            'metric': 'l2',
            'verbose': -1
        }
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
        evals_result = {}
        gbm = lgb.train(params, lgb_train,
                        num_boost_round=50,
                        valid_sets=lgb_eval,
                        verbose_eval=False,
                        evals_result=evals_result)
        ret = mean_squared_error(y_test, gbm.predict(X_test))
        self.assertLess(ret, 16)
        self.assertAlmostEqual(evals_result['valid_0']['l2'][-1], ret, places=5)

    def test_missing_value_handle(self):
        X_train = np.zeros((1000, 1))
        y_train = np.zeros(1000)
        trues = random.sample(range(1000), 200)
        for idx in trues:
            X_train[idx, 0] = np.nan
            y_train[idx] = 1
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_train, y_train)

        params = {
            'metric': 'l2',
            'verbose': -1,
            'boost_from_average': False
        }
        evals_result = {}
        gbm = lgb.train(params, lgb_train,
                        num_boost_round=20,
                        valid_sets=lgb_eval,
                        verbose_eval=True,
                        evals_result=evals_result)
        ret = mean_squared_error(y_train, gbm.predict(X_train))
        self.assertLess(ret, 0.005)
        self.assertAlmostEqual(evals_result['valid_0']['l2'][-1], ret, places=5)

    def test_missing_value_handle_na(self):
        x = [0, 1, 2, 3, 4, 5, 6, 7, np.nan]
        y = [1, 1, 1, 1, 0, 0, 0, 0, 1]

        X_train = np.array(x).reshape(len(x), 1)
        y_train = np.array(y)
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_train, y_train)

        params = {
            'objective': 'regression',
            'metric': 'auc',
            'verbose': -1,
            'boost_from_average': False,
            'min_data': 1,
            'num_leaves': 2,
            'learning_rate': 1,
            'min_data_in_bin': 1,
            'zero_as_missing': False
        }
        evals_result = {}
        gbm = lgb.train(params, lgb_train,
                        num_boost_round=1,
                        valid_sets=lgb_eval,
                        verbose_eval=True,
                        evals_result=evals_result)
        pred = gbm.predict(X_train)
        np.testing.assert_allclose(pred, y)
        ret = roc_auc_score(y_train, pred)
        self.assertGreater(ret, 0.999)
        self.assertAlmostEqual(evals_result['valid_0']['auc'][-1], ret, places=5)

    def test_missing_value_handle_zero(self):
        x = [0, 1, 2, 3, 4, 5, 6, 7, np.nan]
        y = [0, 1, 1, 1, 0, 0, 0, 0, 0]

        X_train = np.array(x).reshape(len(x), 1)
        y_train = np.array(y)
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_train, y_train)

        params = {
            'objective': 'regression',
            'metric': 'auc',
            'verbose': -1,
            'boost_from_average': False,
            'min_data': 1,
            'num_leaves': 2,
            'learning_rate': 1,
            'min_data_in_bin': 1,
            'zero_as_missing': True
        }
        evals_result = {}
        gbm = lgb.train(params, lgb_train,
                        num_boost_round=1,
                        valid_sets=lgb_eval,
                        verbose_eval=True,
                        evals_result=evals_result)
        pred = gbm.predict(X_train)
        np.testing.assert_allclose(pred, y)
        ret = roc_auc_score(y_train, pred)
        self.assertGreater(ret, 0.999)
        self.assertAlmostEqual(evals_result['valid_0']['auc'][-1], ret, places=5)

    def test_missing_value_handle_none(self):
        x = [0, 1, 2, 3, 4, 5, 6, 7, np.nan]
        y = [0, 1, 1, 1, 0, 0, 0, 0, 0]

        X_train = np.array(x).reshape(len(x), 1)
        y_train = np.array(y)
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_train, y_train)

        params = {
            'objective': 'regression',
            'metric': 'auc',
            'verbose': -1,
            'boost_from_average': False,
            'min_data': 1,
            'num_leaves': 2,
            'learning_rate': 1,
            'min_data_in_bin': 1,
            'use_missing': False
        }
        evals_result = {}
        gbm = lgb.train(params, lgb_train,
                        num_boost_round=1,
                        valid_sets=lgb_eval,
                        verbose_eval=True,
                        evals_result=evals_result)
        pred = gbm.predict(X_train)
        self.assertAlmostEqual(pred[0], pred[1], places=5)
        self.assertAlmostEqual(pred[-1], pred[0], places=5)
        ret = roc_auc_score(y_train, pred)
        self.assertGreater(ret, 0.83)
        self.assertAlmostEqual(evals_result['valid_0']['auc'][-1], ret, places=5)

    def test_categorical_handle(self):
        x = [0, 1, 2, 3, 4, 5, 6, 7]
        y = [0, 1, 0, 1, 0, 1, 0, 1]

        X_train = np.array(x).reshape(len(x), 1)
        y_train = np.array(y)
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_train, y_train)

        params = {
            'objective': 'regression',
            'metric': 'auc',
            'verbose': -1,
            'boost_from_average': False,
            'min_data': 1,
            'num_leaves': 2,
            'learning_rate': 1,
            'min_data_in_bin': 1,
            'min_data_per_group': 1,
            'cat_smooth': 1,
            'cat_l2': 0,
            'max_cat_to_onehot': 1,
            'zero_as_missing': True,
            'categorical_column': 0
        }
        evals_result = {}
        gbm = lgb.train(params, lgb_train,
                        num_boost_round=1,
                        valid_sets=lgb_eval,
                        verbose_eval=True,
                        evals_result=evals_result)
        pred = gbm.predict(X_train)
        np.testing.assert_allclose(pred, y)
        ret = roc_auc_score(y_train, pred)
        self.assertGreater(ret, 0.999)
        self.assertAlmostEqual(evals_result['valid_0']['auc'][-1], ret, places=5)

    def test_categorical_handle_na(self):
        x = [0, np.nan, 0, np.nan, 0, np.nan]
        y = [0, 1, 0, 1, 0, 1]

        X_train = np.array(x).reshape(len(x), 1)
        y_train = np.array(y)
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_train, y_train)

        params = {
            'objective': 'regression',
            'metric': 'auc',
            'verbose': -1,
            'boost_from_average': False,
            'min_data': 1,
            'num_leaves': 2,
            'learning_rate': 1,
            'min_data_in_bin': 1,
            'min_data_per_group': 1,
            'cat_smooth': 1,
            'cat_l2': 0,
            'max_cat_to_onehot': 1,
            'zero_as_missing': False,
            'categorical_column': 0
        }
        evals_result = {}
        gbm = lgb.train(params, lgb_train,
                        num_boost_round=1,
                        valid_sets=lgb_eval,
                        verbose_eval=True,
                        evals_result=evals_result)
        pred = gbm.predict(X_train)
        np.testing.assert_allclose(pred, y)
        ret = roc_auc_score(y_train, pred)
        self.assertGreater(ret, 0.999)
        self.assertAlmostEqual(evals_result['valid_0']['auc'][-1], ret, places=5)

    def test_multiclass(self):
        X, y = load_digits(10, True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        params = {
            'objective': 'multiclass',
            'metric': 'multi_logloss',
            'num_class': 10,
            'verbose': -1
        }
        lgb_train = lgb.Dataset(X_train, y_train, params=params)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train, params=params)
        evals_result = {}
        gbm = lgb.train(params, lgb_train,
                        num_boost_round=50,
                        valid_sets=lgb_eval,
                        verbose_eval=False,
                        evals_result=evals_result)
        ret = multi_logloss(y_test, gbm.predict(X_test))
        self.assertLess(ret, 0.2)
        self.assertAlmostEqual(evals_result['valid_0']['multi_logloss'][-1], ret, places=5)

    def test_multiclass_rf(self):
        X, y = load_digits(10, True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        params = {
            'boosting_type': 'rf',
            'objective': 'multiclass',
            'metric': 'multi_logloss',
            'bagging_freq': 1,
            'bagging_fraction': 0.6,
            'feature_fraction': 0.6,
            'num_class': 10,
            'num_leaves': 50,
            'min_data': 1,
            'verbose': -1
        }
        lgb_train = lgb.Dataset(X_train, y_train, params=params)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train, params=params)
        evals_result = {}
        gbm = lgb.train(params, lgb_train,
                        num_boost_round=100,
                        valid_sets=lgb_eval,
                        verbose_eval=False,
                        evals_result=evals_result)
        ret = multi_logloss(y_test, gbm.predict(X_test))
        self.assertLess(ret, 0.4)
        self.assertAlmostEqual(evals_result['valid_0']['multi_logloss'][-1], ret, places=5)

    def test_multiclass_prediction_early_stopping(self):
        X, y = load_digits(10, True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        params = {
            'objective': 'multiclass',
            'metric': 'multi_logloss',
            'num_class': 10,
            'verbose': -1
        }
        lgb_train = lgb.Dataset(X_train, y_train, params=params)
        gbm = lgb.train(params, lgb_train,
                        num_boost_round=50)

        pred_parameter = {"pred_early_stop": True,
                          "pred_early_stop_freq": 5,
                          "pred_early_stop_margin": 1.5}
        ret = multi_logloss(y_test, gbm.predict(X_test, **pred_parameter))
        self.assertLess(ret, 0.8)
        self.assertGreater(ret, 0.5)  # loss will be higher than when evaluating the full model

        pred_parameter = {"pred_early_stop": True,
                          "pred_early_stop_freq": 5,
                          "pred_early_stop_margin": 5.5}
        ret = multi_logloss(y_test, gbm.predict(X_test, **pred_parameter))
        self.assertLess(ret, 0.2)

    def test_multi_class_error(self):
        X, y = load_digits(10, True)
        params = {'objective': 'multiclass', 'num_classes': 10, 'metric': 'multi_error',
                  'num_leaves': 4, 'verbose': -1}
        lgb_data = lgb.Dataset(X, label=y)
        est = lgb.train(params, lgb_data, num_boost_round=10)
        predict_default = est.predict(X)
        results = {}
        est = lgb.train(dict(params, multi_error_top_k=1), lgb_data, num_boost_round=10,
                        valid_sets=[lgb_data], evals_result=results, verbose_eval=False)
        predict_1 = est.predict(X)
        # check that default gives same result as k = 1
        np.testing.assert_allclose(predict_1, predict_default)
        # check against independent calculation for k = 1
        err = top_k_error(y, predict_1, 1)
        self.assertAlmostEqual(results['training']['multi_error'][-1], err)
        # check against independent calculation for k = 2
        results = {}
        est = lgb.train(dict(params, multi_error_top_k=2), lgb_data, num_boost_round=10,
                        valid_sets=[lgb_data], evals_result=results, verbose_eval=False)
        predict_2 = est.predict(X)
        err = top_k_error(y, predict_2, 2)
        self.assertAlmostEqual(results['training']['multi_error@2'][-1], err)
        # check against independent calculation for k = 10
        results = {}
        est = lgb.train(dict(params, multi_error_top_k=10), lgb_data, num_boost_round=10,
                        valid_sets=[lgb_data], evals_result=results, verbose_eval=False)
        predict_3 = est.predict(X)
        err = top_k_error(y, predict_3, 10)
        self.assertAlmostEqual(results['training']['multi_error@10'][-1], err)
        # check cases where predictions are equal
        X = np.array([[0, 0], [0, 0]])
        y = np.array([0, 1])
        lgb_data = lgb.Dataset(X, label=y)
        params['num_classes'] = 2
        results = {}
        lgb.train(params, lgb_data, num_boost_round=10,
                  valid_sets=[lgb_data], evals_result=results, verbose_eval=False)
        self.assertAlmostEqual(results['training']['multi_error'][-1], 1)
        results = {}
        lgb.train(dict(params, multi_error_top_k=2), lgb_data, num_boost_round=10,
                  valid_sets=[lgb_data], evals_result=results, verbose_eval=False)
        self.assertAlmostEqual(results['training']['multi_error@2'][-1], 0)

    def test_early_stopping(self):
        X, y = load_breast_cancer(True)
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbose': -1
        }
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
        valid_set_name = 'valid_set'
        # no early stopping
        gbm = lgb.train(params, lgb_train,
                        num_boost_round=10,
                        valid_sets=lgb_eval,
                        valid_names=valid_set_name,
                        verbose_eval=False,
                        early_stopping_rounds=5)
        self.assertEqual(gbm.best_iteration, 10)
        self.assertIn(valid_set_name, gbm.best_score)
        self.assertIn('binary_logloss', gbm.best_score[valid_set_name])
        # early stopping occurs
        gbm = lgb.train(params, lgb_train,
                        valid_sets=lgb_eval,
                        valid_names=valid_set_name,
                        verbose_eval=False,
                        early_stopping_rounds=5)
        self.assertLessEqual(gbm.best_iteration, 100)
        self.assertIn(valid_set_name, gbm.best_score)
        self.assertIn('binary_logloss', gbm.best_score[valid_set_name])

    def test_continue_train(self):
        X, y = load_boston(True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        params = {
            'objective': 'regression',
            'metric': 'l1',
            'verbose': -1
        }
        lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train, free_raw_data=False)
        init_gbm = lgb.train(params, lgb_train, num_boost_round=20)
        model_name = 'model.txt'
        init_gbm.save_model(model_name)
        evals_result = {}
        gbm = lgb.train(params, lgb_train,
                        num_boost_round=30,
                        valid_sets=lgb_eval,
                        verbose_eval=False,
                        # test custom eval metrics
                        feval=(lambda p, d: ('mae', mean_absolute_error(p, d.get_label()), False)),
                        evals_result=evals_result,
                        init_model='model.txt')
        ret = mean_absolute_error(y_test, gbm.predict(X_test))
        self.assertLess(ret, 3.5)
        self.assertAlmostEqual(evals_result['valid_0']['l1'][-1], ret, places=5)
        for l1, mae in zip(evals_result['valid_0']['l1'], evals_result['valid_0']['mae']):
            self.assertAlmostEqual(l1, mae, places=5)
        os.remove(model_name)

    def test_continue_train_dart(self):
        X, y = load_boston(True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        params = {
            'boosting_type': 'dart',
            'objective': 'regression',
            'metric': 'l1',
            'verbose': -1
        }
        lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train, free_raw_data=False)
        init_gbm = lgb.train(params, lgb_train, num_boost_round=50)
        evals_result = {}
        gbm = lgb.train(params, lgb_train,
                        num_boost_round=50,
                        valid_sets=lgb_eval,
                        verbose_eval=False,
                        evals_result=evals_result,
                        init_model=init_gbm)
        ret = mean_absolute_error(y_test, gbm.predict(X_test))
        self.assertLess(ret, 2.5)
        self.assertAlmostEqual(evals_result['valid_0']['l1'][-1], ret, places=5)

    def test_continue_train_multiclass(self):
        X, y = load_iris(True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        params = {
            'objective': 'multiclass',
            'metric': 'multi_logloss',
            'num_class': 3,
            'verbose': -1
        }
        lgb_train = lgb.Dataset(X_train, y_train, params=params, free_raw_data=False)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train, params=params, free_raw_data=False)
        init_gbm = lgb.train(params, lgb_train, num_boost_round=20)
        evals_result = {}
        gbm = lgb.train(params, lgb_train,
                        num_boost_round=30,
                        valid_sets=lgb_eval,
                        verbose_eval=False,
                        evals_result=evals_result,
                        init_model=init_gbm)
        ret = multi_logloss(y_test, gbm.predict(X_test))
        self.assertLess(ret, 1.5)
        self.assertAlmostEqual(evals_result['valid_0']['multi_logloss'][-1], ret, places=5)

    def test_cv(self):
        X, y = load_boston(True)
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.1, random_state=42)
        params = {'verbose': -1}
        lgb_train = lgb.Dataset(X_train, y_train)
        # shuffle = False, override metric in params
        params_with_metric = {'metric': 'l2', 'verbose': -1}
        cv_res = lgb.cv(params_with_metric, lgb_train, num_boost_round=10,
                        nfold=3, stratified=False, shuffle=False,
                        metrics='l1', verbose_eval=False)
        self.assertIn('l1-mean', cv_res)
        self.assertNotIn('l2-mean', cv_res)
        self.assertEqual(len(cv_res['l1-mean']), 10)
        # shuffle = True, callbacks
        cv_res = lgb.cv(params, lgb_train, num_boost_round=10, nfold=3, stratified=False, shuffle=True,
                        metrics='l1', verbose_eval=False,
                        callbacks=[lgb.reset_parameter(learning_rate=lambda i: 0.1 - 0.001 * i)])
        self.assertIn('l1-mean', cv_res)
        self.assertEqual(len(cv_res['l1-mean']), 10)
        # enable display training loss
        cv_res = lgb.cv(params_with_metric, lgb_train, num_boost_round=10,
                        nfold=3, stratified=False, shuffle=False,
                        metrics='l1', verbose_eval=False, eval_train_metric=True)
        self.assertIn('train l1-mean', cv_res)
        self.assertIn('valid l1-mean', cv_res)
        self.assertNotIn('train l2-mean', cv_res)
        self.assertNotIn('valid l2-mean', cv_res)
        self.assertEqual(len(cv_res['train l1-mean']), 10)
        self.assertEqual(len(cv_res['valid l1-mean']), 10)
        # self defined folds
        tss = TimeSeriesSplit(3)
        folds = tss.split(X_train)
        cv_res_gen = lgb.cv(params_with_metric, lgb_train, num_boost_round=10, folds=folds,
                            verbose_eval=False)
        cv_res_obj = lgb.cv(params_with_metric, lgb_train, num_boost_round=10, folds=tss,
                            verbose_eval=False)
        np.testing.assert_allclose(cv_res_gen['l2-mean'], cv_res_obj['l2-mean'])
        # lambdarank
        X_train, y_train = load_svmlight_file(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                           '../../examples/lambdarank/rank.train'))
        q_train = np.loadtxt(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                          '../../examples/lambdarank/rank.train.query'))
        params_lambdarank = {'objective': 'lambdarank', 'verbose': -1, 'eval_at': 3}
        lgb_train = lgb.Dataset(X_train, y_train, group=q_train)
        # ... with l2 metric
        cv_res_lambda = lgb.cv(params_lambdarank, lgb_train, num_boost_round=10, nfold=3,
                               metrics='l2', verbose_eval=False)
        self.assertEqual(len(cv_res_lambda), 2)
        self.assertFalse(np.isnan(cv_res_lambda['l2-mean']).any())
        # ... with NDCG (default) metric
        cv_res_lambda = lgb.cv(params_lambdarank, lgb_train, num_boost_round=10, nfold=3,
                               verbose_eval=False)
        self.assertEqual(len(cv_res_lambda), 2)
        self.assertFalse(np.isnan(cv_res_lambda['ndcg@3-mean']).any())
        # self defined folds with lambdarank
        cv_res_lambda_obj = lgb.cv(params_lambdarank, lgb_train, num_boost_round=10,
                                   folds=GroupKFold(n_splits=3),
                                   verbose_eval=False)
        np.testing.assert_allclose(cv_res_lambda['ndcg@3-mean'], cv_res_lambda_obj['ndcg@3-mean'])

    def test_feature_name(self):
        X, y = load_boston(True)
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.1, random_state=42)
        params = {'verbose': -1}
        lgb_train = lgb.Dataset(X_train, y_train)
        feature_names = ['f_' + str(i) for i in range(X_train.shape[-1])]
        gbm = lgb.train(params, lgb_train, num_boost_round=5, feature_name=feature_names)
        self.assertListEqual(feature_names, gbm.feature_name())
        # test feature_names with whitespaces
        feature_names_with_space = ['f ' + str(i) for i in range(X_train.shape[-1])]
        gbm = lgb.train(params, lgb_train, num_boost_round=5, feature_name=feature_names_with_space)
        self.assertListEqual(feature_names, gbm.feature_name())

    def test_save_load_copy_pickle(self):
        def test_template(init_model=None, return_model=False):
            X, y = load_boston(True)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
            params = {
                'objective': 'regression',
                'metric': 'l2',
                'verbose': -1
            }
            lgb_train = lgb.Dataset(X_train, y_train)
            gbm_template = lgb.train(params, lgb_train, num_boost_round=10, init_model=init_model)
            return gbm_template if return_model else mean_squared_error(y_test, gbm_template.predict(X_test))
        gbm = test_template(return_model=True)
        ret_origin = test_template(init_model=gbm)
        other_ret = []
        gbm.save_model('lgb.model')
        other_ret.append(test_template(init_model='lgb.model'))
        gbm_load = lgb.Booster(model_file='lgb.model')
        other_ret.append(test_template(init_model=gbm_load))
        other_ret.append(test_template(init_model=copy.copy(gbm)))
        other_ret.append(test_template(init_model=copy.deepcopy(gbm)))
        with open('lgb.pkl', 'wb') as f:
            pickle.dump(gbm, f)
        with open('lgb.pkl', 'rb') as f:
            gbm_pickle = pickle.load(f)
        other_ret.append(test_template(init_model=gbm_pickle))
        gbm_pickles = pickle.loads(pickle.dumps(gbm))
        other_ret.append(test_template(init_model=gbm_pickles))
        for ret in other_ret:
            self.assertAlmostEqual(ret_origin, ret, places=5)

    @unittest.skipIf(not lgb.compat.PANDAS_INSTALLED, 'pandas is not installed')
    def test_pandas_categorical(self):
        import pandas as pd
        np.random.seed(42)  # sometimes there is no difference how cols are treated (cat or not cat)
        X = pd.DataFrame({"A": np.random.permutation(['a', 'b', 'c', 'd'] * 75),  # str
                          "B": np.random.permutation([1, 2, 3] * 100),  # int
                          "C": np.random.permutation([0.1, 0.2, -0.1, -0.1, 0.2] * 60),  # float
                          "D": np.random.permutation([True, False] * 150),  # bool
                          "E": pd.Categorical(np.random.permutation(['z', 'y', 'x', 'w', 'v'] * 60),
                                              ordered=True)})  # str and ordered categorical
        y = np.random.permutation([0, 1] * 150)
        X_test = pd.DataFrame({"A": np.random.permutation(['a', 'b', 'e'] * 20),  # unseen category
                               "B": np.random.permutation([1, 3] * 30),
                               "C": np.random.permutation([0.1, -0.1, 0.2, 0.2] * 15),
                               "D": np.random.permutation([True, False] * 30),
                               "E": pd.Categorical(pd.np.random.permutation(['z', 'y'] * 30),
                                                   ordered=True)})
        np.random.seed()  # reset seed
        cat_cols_actual = ["A", "B", "C", "D"]
        cat_cols_to_store = cat_cols_actual + ["E"]
        X[cat_cols_actual] = X[cat_cols_actual].astype('category')
        X_test[cat_cols_actual] = X_test[cat_cols_actual].astype('category')
        cat_values = [X[col].cat.categories.tolist() for col in cat_cols_to_store]
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbose': -1
        }
        lgb_train = lgb.Dataset(X, y)
        gbm0 = lgb.train(params, lgb_train, num_boost_round=10)
        pred0 = gbm0.predict(X_test)
        self.assertEqual(lgb_train.categorical_feature, 'auto')
        lgb_train = lgb.Dataset(X, pd.DataFrame(y))  # also test that label can be one-column pd.DataFrame
        gbm1 = lgb.train(params, lgb_train, num_boost_round=10, categorical_feature=[0])
        pred1 = gbm1.predict(X_test)
        self.assertListEqual(lgb_train.categorical_feature, [0])
        lgb_train = lgb.Dataset(X, pd.Series(y))  # also test that label can be pd.Series
        gbm2 = lgb.train(params, lgb_train, num_boost_round=10, categorical_feature=['A'])
        pred2 = gbm2.predict(X_test)
        self.assertListEqual(lgb_train.categorical_feature, ['A'])
        lgb_train = lgb.Dataset(X, y)
        gbm3 = lgb.train(params, lgb_train, num_boost_round=10, categorical_feature=['A', 'B', 'C', 'D'])
        pred3 = gbm3.predict(X_test)
        self.assertListEqual(lgb_train.categorical_feature, ['A', 'B', 'C', 'D'])
        gbm3.save_model('categorical.model')
        gbm4 = lgb.Booster(model_file='categorical.model')
        pred4 = gbm4.predict(X_test)
        self.assertListEqual(lgb_train.categorical_feature, ['A', 'B', 'C', 'D'])
        model_str = gbm4.model_to_string()
        gbm4.model_from_string(model_str, False)
        pred5 = gbm4.predict(X_test)
        gbm5 = lgb.Booster(model_str=model_str)
        pred6 = gbm5.predict(X_test)
        lgb_train = lgb.Dataset(X, y)
        gbm6 = lgb.train(params, lgb_train, num_boost_round=10, categorical_feature=['E'])
        pred7 = gbm6.predict(X_test)
        self.assertListEqual(lgb_train.categorical_feature, ['E'])
        lgb_train = lgb.Dataset(X, y)
        gbm7 = lgb.train(params, lgb_train, num_boost_round=10, categorical_feature=[])
        pred8 = gbm7.predict(X_test)
        self.assertListEqual(lgb_train.categorical_feature, [])
        self.assertRaises(AssertionError,
                          np.testing.assert_allclose,
                          pred0, pred1)
        self.assertRaises(AssertionError,
                          np.testing.assert_allclose,
                          pred0, pred2)
        np.testing.assert_allclose(pred1, pred2)
        np.testing.assert_allclose(pred0, pred3)
        np.testing.assert_allclose(pred0, pred4)
        np.testing.assert_allclose(pred0, pred5)
        np.testing.assert_allclose(pred0, pred6)
        self.assertRaises(AssertionError,
                          np.testing.assert_allclose,
                          pred0, pred7)  # ordered cat features aren't treated as cat features by default
        self.assertRaises(AssertionError,
                          np.testing.assert_allclose,
                          pred0, pred8)
        self.assertListEqual(gbm0.pandas_categorical, cat_values)
        self.assertListEqual(gbm1.pandas_categorical, cat_values)
        self.assertListEqual(gbm2.pandas_categorical, cat_values)
        self.assertListEqual(gbm3.pandas_categorical, cat_values)
        self.assertListEqual(gbm4.pandas_categorical, cat_values)
        self.assertListEqual(gbm5.pandas_categorical, cat_values)
        self.assertListEqual(gbm6.pandas_categorical, cat_values)
        self.assertListEqual(gbm7.pandas_categorical, cat_values)

    @unittest.skipIf(not lgb.compat.PANDAS_INSTALLED, 'pandas is not installed')
    def test_pandas_sparse(self):
        import pandas as pd
        X = pd.DataFrame({"A": pd.SparseArray(np.random.permutation([0, 1, 2] * 100)),
                          "B": pd.SparseArray(np.random.permutation([0.0, 0.1, 0.2, -0.1, 0.2] * 60)),
                          "C": pd.SparseArray(np.random.permutation([True, False] * 150))})
        y = pd.Series(pd.SparseArray(np.random.permutation([0, 1] * 150)))
        X_test = pd.DataFrame({"A": pd.SparseArray(np.random.permutation([0, 2] * 30)),
                               "B": pd.SparseArray(np.random.permutation([0.0, 0.1, 0.2, -0.1] * 15)),
                               "C": pd.SparseArray(np.random.permutation([True, False] * 30))})
        if pd.__version__ >= '0.24.0':
            for dtype in pd.concat([X.dtypes, X_test.dtypes, pd.Series(y.dtypes)]):
                self.assertTrue(pd.api.types.is_sparse(dtype))
        params = {
            'objective': 'binary',
            'verbose': -1
        }
        lgb_train = lgb.Dataset(X, y)
        gbm = lgb.train(params, lgb_train, num_boost_round=10)
        pred_sparse = gbm.predict(X_test, raw_score=True)
        if hasattr(X_test, 'sparse'):
            pred_dense = gbm.predict(X_test.sparse.to_dense(), raw_score=True)
        else:
            pred_dense = gbm.predict(X_test.to_dense(), raw_score=True)
        np.testing.assert_allclose(pred_sparse, pred_dense)

    def test_reference_chain(self):
        X = np.random.normal(size=(100, 2))
        y = np.random.normal(size=100)
        tmp_dat = lgb.Dataset(X, y)
        # take subsets and train
        tmp_dat_train = tmp_dat.subset(np.arange(80))
        tmp_dat_val = tmp_dat.subset(np.arange(80, 100)).subset(np.arange(18))
        params = {'objective': 'regression_l2', 'metric': 'rmse'}
        evals_result = {}
        gbm = lgb.train(params, tmp_dat_train, num_boost_round=20,
                        valid_sets=[tmp_dat_train, tmp_dat_val], evals_result=evals_result)
        self.assertEqual(len(evals_result['training']['rmse']), 20)
        self.assertEqual(len(evals_result['valid_1']['rmse']), 20)

    def test_contribs(self):
        X, y = load_breast_cancer(True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbose': -1,
        }
        lgb_train = lgb.Dataset(X_train, y_train)
        gbm = lgb.train(params, lgb_train,
                        num_boost_round=20)

        self.assertLess(np.linalg.norm(gbm.predict(X_test, raw_score=True)
                                       - np.sum(gbm.predict(X_test, pred_contrib=True), axis=1)), 1e-4)

    def test_sliced_data(self):
        def train_and_get_predictions(features, labels):
            dataset = lgb.Dataset(features, label=labels)
            lgb_params = {
                'application': 'binary',
                'verbose': -1,
                'min_data': 5,
            }
            gbm = lgb.train(
                params=lgb_params,
                train_set=dataset,
                num_boost_round=10,
            )
            return gbm.predict(features)

        num_samples = 100
        features = np.random.rand(num_samples, 5)
        positive_samples = int(num_samples * 0.25)
        labels = np.append(
            np.ones(positive_samples, dtype=np.float32),
            np.zeros(num_samples - positive_samples, dtype=np.float32),
        )
        # test sliced labels
        origin_pred = train_and_get_predictions(features, labels)
        stacked_labels = np.column_stack((labels, np.ones(num_samples, dtype=np.float32)))
        sliced_labels = stacked_labels[:, 0]
        sliced_pred = train_and_get_predictions(features, sliced_labels)
        np.testing.assert_allclose(origin_pred, sliced_pred)
        # append some columns
        stacked_features = np.column_stack((np.ones(num_samples, dtype=np.float32), features))
        stacked_features = np.column_stack((np.ones(num_samples, dtype=np.float32), stacked_features))
        stacked_features = np.column_stack((stacked_features, np.ones(num_samples, dtype=np.float32)))
        stacked_features = np.column_stack((stacked_features, np.ones(num_samples, dtype=np.float32)))
        # append some rows
        stacked_features = np.concatenate((np.ones(9, dtype=np.float32).reshape((1, 9)), stacked_features), axis=0)
        stacked_features = np.concatenate((np.ones(9, dtype=np.float32).reshape((1, 9)), stacked_features), axis=0)
        stacked_features = np.concatenate((stacked_features, np.ones(9, dtype=np.float32).reshape((1, 9))), axis=0)
        stacked_features = np.concatenate((stacked_features, np.ones(9, dtype=np.float32).reshape((1, 9))), axis=0)
        # test sliced 2d matrix
        sliced_features = stacked_features[2:102, 2:7]
        self.assertTrue(np.all(sliced_features == features))
        sliced_pred = train_and_get_predictions(sliced_features, sliced_labels)
        np.testing.assert_allclose(origin_pred, sliced_pred)
        # test sliced CSR
        stacked_csr = csr_matrix(stacked_features)
        sliced_csr = stacked_csr[2:102, 2:7]
        self.assertTrue(np.all(sliced_csr == features))
        sliced_pred = train_and_get_predictions(sliced_csr, sliced_labels)
        np.testing.assert_allclose(origin_pred, sliced_pred)

    def test_init_with_subset(self):
        data = np.random.random((500, 2))
        y = [1] * 250 + [0] * 250
        lgb_train = lgb.Dataset(data, y, free_raw_data=False)
        subset_index_1 = np.random.choice(np.arange(500), 300, replace=False)
        subset_data_1 = lgb_train.subset(subset_index_1)
        subset_index_2 = np.random.choice(np.arange(500), 200, replace=False)
        subset_data_2 = lgb_train.subset(subset_index_2)
        params = {
            'objective': 'binary',
            'verbose': -1
        }
        init_gbm = lgb.train(params=params,
                             train_set=subset_data_1,
                             num_boost_round=10,
                             keep_training_booster=True)
        gbm = lgb.train(params=params,
                        train_set=subset_data_2,
                        num_boost_round=10,
                        init_model=init_gbm)
        self.assertEqual(lgb_train.get_data().shape[0], 500)
        self.assertEqual(subset_data_1.get_data().shape[0], 300)
        self.assertEqual(subset_data_2.get_data().shape[0], 200)
        lgb_train.save_binary("lgb_train_data.bin")
        lgb_train_from_file = lgb.Dataset('lgb_train_data.bin', free_raw_data=False)
        subset_data_3 = lgb_train_from_file.subset(subset_index_1)
        subset_data_4 = lgb_train_from_file.subset(subset_index_2)
        init_gbm_2 = lgb.train(params=params,
                               train_set=subset_data_3,
                               num_boost_round=10,
                               keep_training_booster=True)
        with np.testing.assert_raises_regex(lgb.basic.LightGBMError, "Unknown format of training data"):
            gbm = lgb.train(params=params,
                            train_set=subset_data_4,
                            num_boost_round=10,
                            init_model=init_gbm_2)
        self.assertEqual(lgb_train_from_file.get_data(), "lgb_train_data.bin")
        self.assertEqual(subset_data_3.get_data(), "lgb_train_data.bin")
        self.assertEqual(subset_data_4.get_data(), "lgb_train_data.bin")

    def test_monotone_constraint(self):
        def is_increasing(y):
            return (np.diff(y) >= 0.0).all()

        def is_decreasing(y):
            return (np.diff(y) <= 0.0).all()

        def is_correctly_constrained(learner):
            n = 200
            variable_x = np.linspace(0, 1, n).reshape((n, 1))
            fixed_xs_values = np.linspace(0, 1, n)
            for i in range(n):
                fixed_x = fixed_xs_values[i] * np.ones((n, 1))
                monotonically_increasing_x = np.column_stack((variable_x, fixed_x))
                monotonically_increasing_y = learner.predict(monotonically_increasing_x)
                monotonically_decreasing_x = np.column_stack((fixed_x, variable_x))
                monotonically_decreasing_y = learner.predict(monotonically_decreasing_x)
                if not (is_increasing(monotonically_increasing_y) and is_decreasing(monotonically_decreasing_y)):
                    return False
            return True

        number_of_dpoints = 3000
        x1_positively_correlated_with_y = np.random.random(size=number_of_dpoints)
        x2_negatively_correlated_with_y = np.random.random(size=number_of_dpoints)
        x = np.column_stack((x1_positively_correlated_with_y, x2_negatively_correlated_with_y))
        zs = np.random.normal(loc=0.0, scale=0.01, size=number_of_dpoints)
        y = (5 * x1_positively_correlated_with_y
             + np.sin(10 * np.pi * x1_positively_correlated_with_y)
             - 5 * x2_negatively_correlated_with_y
             - np.cos(10 * np.pi * x2_negatively_correlated_with_y)
             + zs)
        trainset = lgb.Dataset(x, label=y)
        params = {
            'min_data': 20,
            'num_leaves': 20,
            'monotone_constraints': '1,-1'
        }
        constrained_model = lgb.train(params, trainset)
        self.assertTrue(is_correctly_constrained(constrained_model))

    def test_max_bin_by_feature(self):
        col1 = np.arange(0, 100)[:, np.newaxis]
        col2 = np.zeros((100, 1))
        col2[20:] = 1
        X = np.concatenate([col1, col2], axis=1)
        y = np.arange(0, 100)
        params = {
            'objective': 'regression_l2',
            'verbose': -1,
            'num_leaves': 100,
            'min_data_in_leaf': 1,
            'min_sum_hessian_in_leaf': 0,
            'min_data_in_bin': 1,
            'max_bin_by_feature': [100, 2]
        }
        lgb_data = lgb.Dataset(X, label=y)
        est = lgb.train(params, lgb_data, num_boost_round=1)
        self.assertEqual(len(np.unique(est.predict(X))), 100)
        params['max_bin_by_feature'] = [2, 100]
        lgb_data = lgb.Dataset(X, label=y)
        est = lgb.train(params, lgb_data, num_boost_round=1)
        self.assertEqual(len(np.unique(est.predict(X))), 3)

    def test_small_max_bin(self):
        np.random.seed(0)
        y = np.random.choice([0, 1], 100)
        x = np.zeros((100, 1))
        x[:30, 0] = -1
        x[30:60, 0] = 1
        x[60:, 0] = 2
        params = {'objective': 'binary',
                  'seed': 0,
                  'min_data_in_leaf': 1,
                  'verbose': -1,
                  'max_bin': 2}
        lgb_x = lgb.Dataset(x, label=y)
        est = lgb.train(params, lgb_x, num_boost_round=5)
        x[0, 0] = np.nan
        params['max_bin'] = 3
        lgb_x = lgb.Dataset(x, label=y)
        est = lgb.train(params, lgb_x, num_boost_round=5)
        np.random.seed()  # reset seed

    def test_refit(self):
        X, y = load_breast_cancer(True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbose': -1,
            'min_data': 10
        }
        lgb_train = lgb.Dataset(X_train, y_train)
        gbm = lgb.train(params, lgb_train, num_boost_round=20)
        err_pred = log_loss(y_test, gbm.predict(X_test))
        new_gbm = gbm.refit(X_test, y_test)
        new_err_pred = log_loss(y_test, new_gbm.predict(X_test))
        self.assertGreater(err_pred, new_err_pred)

    def test_mape_rf(self):
        X, y = load_boston(True)
        params = {
            'boosting_type': 'rf',
            'objective': 'mape',
            'verbose': -1,
            'bagging_freq': 1,
            'bagging_fraction': 0.8,
            'feature_fraction': 0.8,
            'boost_from_average': True
        }
        lgb_train = lgb.Dataset(X, y)
        gbm = lgb.train(params, lgb_train, num_boost_round=20)
        pred = gbm.predict(X)
        pred_mean = pred.mean()
        self.assertGreater(pred_mean, 20)

    def test_mape_dart(self):
        X, y = load_boston(True)
        params = {
            'boosting_type': 'dart',
            'objective': 'mape',
            'verbose': -1,
            'bagging_freq': 1,
            'bagging_fraction': 0.8,
            'feature_fraction': 0.8,
            'boost_from_average': False
        }
        lgb_train = lgb.Dataset(X, y)
        gbm = lgb.train(params, lgb_train, num_boost_round=40)
        pred = gbm.predict(X)
        pred_mean = pred.mean()
        self.assertGreater(pred_mean, 18)

    def check_constant_features(self, y_true, expected_pred, more_params):
        X_train = np.ones((len(y_true), 1))
        y_train = np.array(y_true)
        params = {
            'objective': 'regression',
            'num_class': 1,
            'verbose': -1,
            'min_data': 1,
            'num_leaves': 2,
            'learning_rate': 1,
            'min_data_in_bin': 1,
            'boost_from_average': True
        }
        params.update(more_params)
        lgb_train = lgb.Dataset(X_train, y_train, params=params)
        gbm = lgb.train(params, lgb_train, num_boost_round=2)
        pred = gbm.predict(X_train)
        self.assertTrue(np.allclose(pred, expected_pred))

    def test_constant_features_regression(self):
        params = {
            'objective': 'regression'
        }
        self.check_constant_features([0.0, 10.0, 0.0, 10.0], 5.0, params)
        self.check_constant_features([0.0, 1.0, 2.0, 3.0], 1.5, params)
        self.check_constant_features([-1.0, 1.0, -2.0, 2.0], 0.0, params)

    def test_constant_features_binary(self):
        params = {
            'objective': 'binary'
        }
        self.check_constant_features([0.0, 10.0, 0.0, 10.0], 0.5, params)
        self.check_constant_features([0.0, 1.0, 2.0, 3.0], 0.75, params)

    def test_constant_features_multiclass(self):
        params = {
            'objective': 'multiclass',
            'num_class': 3
        }
        self.check_constant_features([0.0, 1.0, 2.0, 0.0], [0.5, 0.25, 0.25], params)
        self.check_constant_features([0.0, 1.0, 2.0, 1.0], [0.25, 0.5, 0.25], params)

    def test_constant_features_multiclassova(self):
        params = {
            'objective': 'multiclassova',
            'num_class': 3
        }
        self.check_constant_features([0.0, 1.0, 2.0, 0.0], [0.5, 0.25, 0.25], params)
        self.check_constant_features([0.0, 1.0, 2.0, 1.0], [0.25, 0.5, 0.25], params)

    def test_fpreproc(self):
        def preprocess_data(dtrain, dtest, params):
            train_data = dtrain.construct().get_data()
            test_data = dtest.construct().get_data()
            train_data[:, 0] += 1
            test_data[:, 0] += 1
            dtrain.label[-5:] = 3
            dtest.label[-5:] = 3
            dtrain = lgb.Dataset(train_data, dtrain.label)
            dtest = lgb.Dataset(test_data, dtest.label, reference=dtrain)
            params['num_class'] = 4
            return dtrain, dtest, params

        X, y = load_iris(True)
        dataset = lgb.Dataset(X, y, free_raw_data=False)
        params = {'objective': 'multiclass', 'num_class': 3, 'verbose': -1}
        results = lgb.cv(params, dataset, num_boost_round=10, fpreproc=preprocess_data)
        self.assertIn('multi_logloss-mean', results)
        self.assertEqual(len(results['multi_logloss-mean']), 10)

    def test_metrics(self):
        def custom_obj(preds, train_data):
            return np.zeros(preds.shape), np.zeros(preds.shape)

        def custom_metric(preds, train_data):
            return 'error', 0, False

        X, y = load_digits(2, True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        lgb_train = lgb.Dataset(X_train, y_train, silent=True)
        lgb_valid = lgb.Dataset(X_test, y_test, reference=lgb_train, silent=True)

        evals_result = {}
        params_verbose = {'verbose': -1}
        params_obj_verbose = {'objective': 'binary', 'verbose': -1}
        params_obj_metric_log_verbose = {'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1}
        params_obj_metric_err_verbose = {'objective': 'binary', 'metric': 'binary_error', 'verbose': -1}
        params_obj_metric_inv_verbose = {'objective': 'binary', 'metric': 'invalid_metric', 'verbose': -1}
        params_obj_metric_multi_verbose = {'objective': 'binary',
                                           'metric': ['binary_logloss', 'binary_error'],
                                           'verbose': -1}
        params_obj_metric_none_verbose = {'objective': 'binary', 'metric': 'None', 'verbose': -1}
        params_metric_log_verbose = {'metric': 'binary_logloss', 'verbose': -1}
        params_metric_err_verbose = {'metric': 'binary_error', 'verbose': -1}
        params_metric_inv_verbose = {'metric_types': 'invalid_metric', 'verbose': -1}
        params_metric_multi_verbose = {'metric': ['binary_logloss', 'binary_error'], 'verbose': -1}
        params_metric_none_verbose = {'metric': 'None', 'verbose': -1}

        def get_cv_result(params=params_obj_verbose, **kwargs):
            return lgb.cv(params, lgb_train, num_boost_round=5, verbose_eval=False, **kwargs)

        def train_booster(params=params_obj_verbose, **kwargs):
            lgb.train(params, lgb_train,
                      num_boost_round=5,
                      valid_sets=[lgb_valid],
                      evals_result=evals_result,
                      verbose_eval=False, **kwargs)

        # no fobj, no feval
        # default metric
        res = get_cv_result()
        self.assertEqual(len(res), 2)
        self.assertIn('binary_logloss-mean', res)

        # non-default metric in params
        res = get_cv_result(params=params_obj_metric_err_verbose)
        self.assertEqual(len(res), 2)
        self.assertIn('binary_error-mean', res)

        # default metric in args
        res = get_cv_result(metrics='binary_logloss')
        self.assertEqual(len(res), 2)
        self.assertIn('binary_logloss-mean', res)

        # non-default metric in args
        res = get_cv_result(metrics='binary_error')
        self.assertEqual(len(res), 2)
        self.assertIn('binary_error-mean', res)

        # metric in args overwrites one in params
        res = get_cv_result(params=params_obj_metric_inv_verbose, metrics='binary_error')
        self.assertEqual(len(res), 2)
        self.assertIn('binary_error-mean', res)

        # multiple metrics in params
        res = get_cv_result(params=params_obj_metric_multi_verbose)
        self.assertEqual(len(res), 4)
        self.assertIn('binary_logloss-mean', res)
        self.assertIn('binary_error-mean', res)

        # multiple metrics in args
        res = get_cv_result(metrics=['binary_logloss', 'binary_error'])
        self.assertEqual(len(res), 4)
        self.assertIn('binary_logloss-mean', res)
        self.assertIn('binary_error-mean', res)

        # remove default metric by 'None' in list
        res = get_cv_result(metrics=['None'])
        self.assertEqual(len(res), 0)

        # remove default metric by 'None' aliases
        for na_alias in ('None', 'na', 'null', 'custom'):
            res = get_cv_result(metrics=na_alias)
            self.assertEqual(len(res), 0)

        # fobj, no feval
        # no default metric
        res = get_cv_result(params=params_verbose, fobj=custom_obj)
        self.assertEqual(len(res), 0)

        # metric in params
        res = get_cv_result(params=params_metric_err_verbose, fobj=custom_obj)
        self.assertEqual(len(res), 2)
        self.assertIn('binary_error-mean', res)

        # metric in args
        res = get_cv_result(params=params_verbose, fobj=custom_obj, metrics='binary_error')
        self.assertEqual(len(res), 2)
        self.assertIn('binary_error-mean', res)

        # metric in args overwrites its' alias in params
        res = get_cv_result(params=params_metric_inv_verbose, fobj=custom_obj, metrics='binary_error')
        self.assertEqual(len(res), 2)
        self.assertIn('binary_error-mean', res)

        # multiple metrics in params
        res = get_cv_result(params=params_metric_multi_verbose, fobj=custom_obj)
        self.assertEqual(len(res), 4)
        self.assertIn('binary_logloss-mean', res)
        self.assertIn('binary_error-mean', res)

        # multiple metrics in args
        res = get_cv_result(params=params_verbose, fobj=custom_obj,
                            metrics=['binary_logloss', 'binary_error'])
        self.assertEqual(len(res), 4)
        self.assertIn('binary_logloss-mean', res)
        self.assertIn('binary_error-mean', res)

        # no fobj, feval
        # default metric with custom one
        res = get_cv_result(feval=custom_metric)
        self.assertEqual(len(res), 4)
        self.assertIn('binary_logloss-mean', res)
        self.assertIn('error-mean', res)

        # non-default metric in params with custom one
        res = get_cv_result(params=params_obj_metric_err_verbose, feval=custom_metric)
        self.assertEqual(len(res), 4)
        self.assertIn('binary_error-mean', res)
        self.assertIn('error-mean', res)

        # default metric in args with custom one
        res = get_cv_result(metrics='binary_logloss', feval=custom_metric)
        self.assertEqual(len(res), 4)
        self.assertIn('binary_logloss-mean', res)
        self.assertIn('error-mean', res)

        # non-default metric in args with custom one
        res = get_cv_result(metrics='binary_error', feval=custom_metric)
        self.assertEqual(len(res), 4)
        self.assertIn('binary_error-mean', res)
        self.assertIn('error-mean', res)

        # metric in args overwrites one in params, custom one is evaluated too
        res = get_cv_result(params=params_obj_metric_inv_verbose, metrics='binary_error', feval=custom_metric)
        self.assertEqual(len(res), 4)
        self.assertIn('binary_error-mean', res)
        self.assertIn('error-mean', res)

        # multiple metrics in params with custom one
        res = get_cv_result(params=params_obj_metric_multi_verbose, feval=custom_metric)
        self.assertEqual(len(res), 6)
        self.assertIn('binary_logloss-mean', res)
        self.assertIn('binary_error-mean', res)
        self.assertIn('error-mean', res)

        # multiple metrics in args with custom one
        res = get_cv_result(metrics=['binary_logloss', 'binary_error'], feval=custom_metric)
        self.assertEqual(len(res), 6)
        self.assertIn('binary_logloss-mean', res)
        self.assertIn('binary_error-mean', res)
        self.assertIn('error-mean', res)

        # custom metric is evaluated despite 'None' is passed
        res = get_cv_result(metrics=['None'], feval=custom_metric)
        self.assertEqual(len(res), 2)
        self.assertIn('error-mean', res)

        # fobj, feval
        # no default metric, only custom one
        res = get_cv_result(params=params_verbose, fobj=custom_obj, feval=custom_metric)
        self.assertEqual(len(res), 2)
        self.assertIn('error-mean', res)

        # metric in params with custom one
        res = get_cv_result(params=params_metric_err_verbose, fobj=custom_obj, feval=custom_metric)
        self.assertEqual(len(res), 4)
        self.assertIn('binary_error-mean', res)
        self.assertIn('error-mean', res)

        # metric in args with custom one
        res = get_cv_result(params=params_verbose, fobj=custom_obj,
                            feval=custom_metric, metrics='binary_error')
        self.assertEqual(len(res), 4)
        self.assertIn('binary_error-mean', res)
        self.assertIn('error-mean', res)

        # metric in args overwrites one in params, custom one is evaluated too
        res = get_cv_result(params=params_metric_inv_verbose, fobj=custom_obj,
                            feval=custom_metric, metrics='binary_error')
        self.assertEqual(len(res), 4)
        self.assertIn('binary_error-mean', res)
        self.assertIn('error-mean', res)

        # multiple metrics in params with custom one
        res = get_cv_result(params=params_metric_multi_verbose, fobj=custom_obj, feval=custom_metric)
        self.assertEqual(len(res), 6)
        self.assertIn('binary_logloss-mean', res)
        self.assertIn('binary_error-mean', res)
        self.assertIn('error-mean', res)

        # multiple metrics in args with custom one
        res = get_cv_result(params=params_verbose, fobj=custom_obj, feval=custom_metric,
                            metrics=['binary_logloss', 'binary_error'])
        self.assertEqual(len(res), 6)
        self.assertIn('binary_logloss-mean', res)
        self.assertIn('binary_error-mean', res)
        self.assertIn('error-mean', res)

        # custom metric is evaluated despite 'None' is passed
        res = get_cv_result(params=params_metric_none_verbose, fobj=custom_obj, feval=custom_metric)
        self.assertEqual(len(res), 2)
        self.assertIn('error-mean', res)

        # no fobj, no feval
        # default metric
        train_booster()
        self.assertEqual(len(evals_result['valid_0']), 1)
        self.assertIn('binary_logloss', evals_result['valid_0'])

        # default metric in params
        train_booster(params=params_obj_metric_log_verbose)
        self.assertEqual(len(evals_result['valid_0']), 1)
        self.assertIn('binary_logloss', evals_result['valid_0'])

        # non-default metric in params
        train_booster(params=params_obj_metric_err_verbose)
        self.assertEqual(len(evals_result['valid_0']), 1)
        self.assertIn('binary_error', evals_result['valid_0'])

        # multiple metrics in params
        train_booster(params=params_obj_metric_multi_verbose)
        self.assertEqual(len(evals_result['valid_0']), 2)
        self.assertIn('binary_logloss', evals_result['valid_0'])
        self.assertIn('binary_error', evals_result['valid_0'])

        # remove default metric by 'None' aliases
        for na_alias in ('None', 'na', 'null', 'custom'):
            params = {'objective': 'binary', 'metric': na_alias, 'verbose': -1}
            train_booster(params=params)
            self.assertEqual(len(evals_result), 0)

        # fobj, no feval
        # no default metric
        train_booster(params=params_verbose, fobj=custom_obj)
        self.assertEqual(len(evals_result), 0)

        # metric in params
        train_booster(params=params_metric_log_verbose, fobj=custom_obj)
        self.assertEqual(len(evals_result['valid_0']), 1)
        self.assertIn('binary_logloss', evals_result['valid_0'])

        # multiple metrics in params
        train_booster(params=params_metric_multi_verbose, fobj=custom_obj)
        self.assertEqual(len(evals_result['valid_0']), 2)
        self.assertIn('binary_logloss', evals_result['valid_0'])
        self.assertIn('binary_error', evals_result['valid_0'])

        # no fobj, feval
        # default metric with custom one
        train_booster(feval=custom_metric)
        self.assertEqual(len(evals_result['valid_0']), 2)
        self.assertIn('binary_logloss', evals_result['valid_0'])
        self.assertIn('error', evals_result['valid_0'])

        # default metric in params with custom one
        train_booster(params=params_obj_metric_log_verbose, feval=custom_metric)
        self.assertEqual(len(evals_result['valid_0']), 2)
        self.assertIn('binary_logloss', evals_result['valid_0'])
        self.assertIn('error', evals_result['valid_0'])

        # non-default metric in params with custom one
        train_booster(params=params_obj_metric_err_verbose, feval=custom_metric)
        self.assertEqual(len(evals_result['valid_0']), 2)
        self.assertIn('binary_error', evals_result['valid_0'])
        self.assertIn('error', evals_result['valid_0'])

        # multiple metrics in params with custom one
        train_booster(params=params_obj_metric_multi_verbose, feval=custom_metric)
        self.assertEqual(len(evals_result['valid_0']), 3)
        self.assertIn('binary_logloss', evals_result['valid_0'])
        self.assertIn('binary_error', evals_result['valid_0'])
        self.assertIn('error', evals_result['valid_0'])

        # custom metric is evaluated despite 'None' is passed
        train_booster(params=params_obj_metric_none_verbose, feval=custom_metric)
        self.assertEqual(len(evals_result), 1)
        self.assertIn('error', evals_result['valid_0'])

        # fobj, feval
        # no default metric, only custom one
        train_booster(params=params_verbose, fobj=custom_obj, feval=custom_metric)
        self.assertEqual(len(evals_result['valid_0']), 1)
        self.assertIn('error', evals_result['valid_0'])

        # metric in params with custom one
        train_booster(params=params_metric_log_verbose, fobj=custom_obj, feval=custom_metric)
        self.assertEqual(len(evals_result['valid_0']), 2)
        self.assertIn('binary_logloss', evals_result['valid_0'])
        self.assertIn('error', evals_result['valid_0'])

        # multiple metrics in params with custom one
        train_booster(params=params_metric_multi_verbose, fobj=custom_obj, feval=custom_metric)
        self.assertEqual(len(evals_result['valid_0']), 3)
        self.assertIn('binary_logloss', evals_result['valid_0'])
        self.assertIn('binary_error', evals_result['valid_0'])
        self.assertIn('error', evals_result['valid_0'])

        # custom metric is evaluated despite 'None' is passed
        train_booster(params=params_metric_none_verbose, fobj=custom_obj, feval=custom_metric)
        self.assertEqual(len(evals_result), 1)
        self.assertIn('error', evals_result['valid_0'])

        X, y = load_digits(3, True)
        lgb_train = lgb.Dataset(X, y, silent=True)

        obj_multi_aliases = ['multiclass', 'softmax', 'multiclassova', 'multiclass_ova', 'ova', 'ovr']
        for obj_multi_alias in obj_multi_aliases:
            params_obj_class_3_verbose = {'objective': obj_multi_alias, 'num_class': 3, 'verbose': -1}
            params_obj_class_1_verbose = {'objective': obj_multi_alias, 'num_class': 1, 'verbose': -1}
            params_obj_verbose = {'objective': obj_multi_alias, 'verbose': -1}
            # multiclass default metric
            res = get_cv_result(params_obj_class_3_verbose)
            self.assertEqual(len(res), 2)
            self.assertIn('multi_logloss-mean', res)
            # multiclass default metric with custom one
            res = get_cv_result(params_obj_class_3_verbose, feval=custom_metric)
            self.assertEqual(len(res), 4)
            self.assertIn('multi_logloss-mean', res)
            self.assertIn('error-mean', res)
            # multiclass metric alias with custom one for custom objective
            res = get_cv_result(params_obj_class_3_verbose, fobj=custom_obj, feval=custom_metric)
            self.assertEqual(len(res), 2)
            self.assertIn('error-mean', res)
            # no metric for invalid class_num
            res = get_cv_result(params_obj_class_1_verbose, fobj=custom_obj)
            self.assertEqual(len(res), 0)
            # custom metric for invalid class_num
            res = get_cv_result(params_obj_class_1_verbose, fobj=custom_obj, feval=custom_metric)
            self.assertEqual(len(res), 2)
            self.assertIn('error-mean', res)
            # multiclass metric alias with custom one with invalid class_num
            self.assertRaises(lgb.basic.LightGBMError, get_cv_result,
                              params_obj_class_1_verbose, metrics=obj_multi_alias,
                              fobj=custom_obj, feval=custom_metric)
            # multiclass default metric without num_class
            self.assertRaises(lgb.basic.LightGBMError, get_cv_result,
                              params_obj_verbose)
            for metric_multi_alias in obj_multi_aliases + ['multi_logloss']:
                # multiclass metric alias
                res = get_cv_result(params_obj_class_3_verbose, metrics=metric_multi_alias)
                self.assertEqual(len(res), 2)
                self.assertIn('multi_logloss-mean', res)
            # multiclass metric
            res = get_cv_result(params_obj_class_3_verbose, metrics='multi_error')
            self.assertEqual(len(res), 2)
            self.assertIn('multi_error-mean', res)
            # non-valid metric for multiclass objective
            self.assertRaises(lgb.basic.LightGBMError, get_cv_result,
                              params_obj_class_3_verbose, metrics='binary_logloss')
        params_class_3_verbose = {'num_class': 3, 'verbose': -1}
        # non-default num_class for default objective
        self.assertRaises(lgb.basic.LightGBMError, get_cv_result,
                          params_class_3_verbose)
        # no metric with non-default num_class for custom objective
        res = get_cv_result(params_class_3_verbose, fobj=custom_obj)
        self.assertEqual(len(res), 0)
        for metric_multi_alias in obj_multi_aliases + ['multi_logloss']:
            # multiclass metric alias for custom objective
            res = get_cv_result(params_class_3_verbose, metrics=metric_multi_alias, fobj=custom_obj)
            self.assertEqual(len(res), 2)
            self.assertIn('multi_logloss-mean', res)
        # multiclass metric for custom objective
        res = get_cv_result(params_class_3_verbose, metrics='multi_error', fobj=custom_obj)
        self.assertEqual(len(res), 2)
        self.assertIn('multi_error-mean', res)
        # binary metric with non-default num_class for custom objective
        self.assertRaises(lgb.basic.LightGBMError, get_cv_result,
                          params_class_3_verbose, metrics='binary_error', fobj=custom_obj)

    @unittest.skipIf(psutil.virtual_memory().available / 1024 / 1024 / 1024 < 3, 'not enough RAM')
    def test_model_size(self):
        X, y = load_boston(True)
        data = lgb.Dataset(X, y)
        bst = lgb.train({'verbose': -1}, data, num_boost_round=2)
        y_pred = bst.predict(X)
        model_str = bst.model_to_string()
        one_tree = model_str[model_str.find('Tree=1'):model_str.find('end of trees')]
        one_tree_size = len(one_tree)
        one_tree = one_tree.replace('Tree=1', 'Tree={}')
        multiplier = 100
        total_trees = multiplier + 2
        try:
            new_model_str = (model_str[:model_str.find('tree_sizes')]
                             + '\n\n'
                             + model_str[model_str.find('Tree=0'):model_str.find('end of trees')]
                             + (one_tree * multiplier).format(*range(2, total_trees))
                             + model_str[model_str.find('end of trees'):]
                             + ' ' * (2**31 - one_tree_size * total_trees))
            self.assertGreater(len(new_model_str), 2**31)
            bst.model_from_string(new_model_str, verbose=False)
            self.assertEqual(bst.num_trees(), total_trees)
            y_pred_new = bst.predict(X, num_iteration=2)
            np.testing.assert_allclose(y_pred, y_pred_new)
        except MemoryError:
            self.skipTest('not enough RAM')

    def test_get_split_value_histogram(self):
        X, y = load_boston(True)
        lgb_train = lgb.Dataset(X, y, categorical_feature=[2])
        gbm = lgb.train({'verbose': -1}, lgb_train, num_boost_round=20)
        # test XGBoost-style return value
        params = {'feature': 0, 'xgboost_style': True}
        self.assertTupleEqual(gbm.get_split_value_histogram(**params).shape, (9, 2))
        self.assertTupleEqual(gbm.get_split_value_histogram(bins=999, **params).shape, (9, 2))
        self.assertTupleEqual(gbm.get_split_value_histogram(bins=-1, **params).shape, (1, 2))
        self.assertTupleEqual(gbm.get_split_value_histogram(bins=0, **params).shape, (1, 2))
        self.assertTupleEqual(gbm.get_split_value_histogram(bins=1, **params).shape, (1, 2))
        self.assertTupleEqual(gbm.get_split_value_histogram(bins=2, **params).shape, (2, 2))
        self.assertTupleEqual(gbm.get_split_value_histogram(bins=6, **params).shape, (5, 2))
        self.assertTupleEqual(gbm.get_split_value_histogram(bins=7, **params).shape, (6, 2))
        if lgb.compat.PANDAS_INSTALLED:
            np.testing.assert_allclose(
                gbm.get_split_value_histogram(0, xgboost_style=True).values,
                gbm.get_split_value_histogram(gbm.feature_name()[0], xgboost_style=True).values
            )
            np.testing.assert_allclose(
                gbm.get_split_value_histogram(X.shape[-1] - 1, xgboost_style=True).values,
                gbm.get_split_value_histogram(gbm.feature_name()[X.shape[-1] - 1], xgboost_style=True).values
            )
        else:
            np.testing.assert_allclose(
                gbm.get_split_value_histogram(0, xgboost_style=True),
                gbm.get_split_value_histogram(gbm.feature_name()[0], xgboost_style=True)
            )
            np.testing.assert_allclose(
                gbm.get_split_value_histogram(X.shape[-1] - 1, xgboost_style=True),
                gbm.get_split_value_histogram(gbm.feature_name()[X.shape[-1] - 1], xgboost_style=True)
            )
        # test numpy-style return value
        hist, bins = gbm.get_split_value_histogram(0)
        self.assertEqual(len(hist), 23)
        self.assertEqual(len(bins), 24)
        hist, bins = gbm.get_split_value_histogram(0, bins=999)
        self.assertEqual(len(hist), 999)
        self.assertEqual(len(bins), 1000)
        self.assertRaises(ValueError, gbm.get_split_value_histogram, 0, bins=-1)
        self.assertRaises(ValueError, gbm.get_split_value_histogram, 0, bins=0)
        hist, bins = gbm.get_split_value_histogram(0, bins=1)
        self.assertEqual(len(hist), 1)
        self.assertEqual(len(bins), 2)
        hist, bins = gbm.get_split_value_histogram(0, bins=2)
        self.assertEqual(len(hist), 2)
        self.assertEqual(len(bins), 3)
        hist, bins = gbm.get_split_value_histogram(0, bins=6)
        self.assertEqual(len(hist), 6)
        self.assertEqual(len(bins), 7)
        hist, bins = gbm.get_split_value_histogram(0, bins=7)
        self.assertEqual(len(hist), 7)
        self.assertEqual(len(bins), 8)
        hist_idx, bins_idx = gbm.get_split_value_histogram(0)
        hist_name, bins_name = gbm.get_split_value_histogram(gbm.feature_name()[0])
        np.testing.assert_array_equal(hist_idx, hist_name)
        np.testing.assert_allclose(bins_idx, bins_name)
        hist_idx, bins_idx = gbm.get_split_value_histogram(X.shape[-1] - 1)
        hist_name, bins_name = gbm.get_split_value_histogram(gbm.feature_name()[X.shape[-1] - 1])
        np.testing.assert_array_equal(hist_idx, hist_name)
        np.testing.assert_allclose(bins_idx, bins_name)
        # test bins string type
        if np.__version__ > '1.11.0':
            hist_vals, bin_edges = gbm.get_split_value_histogram(0, bins='auto')
            hist = gbm.get_split_value_histogram(0, bins='auto', xgboost_style=True)
            if lgb.compat.PANDAS_INSTALLED:
                mask = hist_vals > 0
                np.testing.assert_array_equal(hist_vals[mask], hist['Count'].values)
                np.testing.assert_allclose(bin_edges[1:][mask], hist['SplitValue'].values)
            else:
                mask = hist_vals > 0
                np.testing.assert_array_equal(hist_vals[mask], hist[:, 1])
                np.testing.assert_allclose(bin_edges[1:][mask], hist[:, 0])
        # test histogram is disabled for categorical features
        self.assertRaises(lgb.basic.LightGBMError, gbm.get_split_value_histogram, 2)

    def test_early_stopping_for_only_first_metric(self):

        def decreasing_metric(preds, train_data):
            return ('decreasing_metric', next(decreasing_generator), False)

        def constant_metric(preds, train_data):
            return ('constant_metric', 0.0, False)

        def metrics_combination_train_regression(valid_sets, metric_list, assumed_iteration,
                                                 first_metric_only, feval=None):
            params = {
                'objective': 'regression',
                'learning_rate': 1.1,
                'num_leaves': 10,
                'metric': metric_list,
                'verbose': -1,
                'seed': 123
            }
            gbm = lgb.train(dict(params, first_metric_only=first_metric_only), lgb_train,
                            num_boost_round=25, valid_sets=valid_sets, feval=feval,
                            early_stopping_rounds=5, verbose_eval=False)
            self.assertEqual(assumed_iteration, gbm.best_iteration)

        def metrics_combination_cv_regression(metric_list, assumed_iteration,
                                              first_metric_only, eval_train_metric, feval=None):
            params = {
                'objective': 'regression',
                'learning_rate': 0.9,
                'num_leaves': 10,
                'metric': metric_list,
                'verbose': -1,
                'seed': 123,
                'gpu_use_dp': True
            }
            ret = lgb.cv(dict(params, first_metric_only=first_metric_only),
                         train_set=lgb_train, num_boost_round=25,
                         stratified=False, feval=feval,
                         early_stopping_rounds=5, verbose_eval=False,
                         eval_train_metric=eval_train_metric)
            self.assertEqual(assumed_iteration, len(ret[list(ret.keys())[0]]))

        decreasing_generator = itertools.count(0, -1)
        X, y = load_boston(True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_test1, X_test2, y_test1, y_test2 = train_test_split(X_test, y_test, test_size=0.5, random_state=73)
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_valid1 = lgb.Dataset(X_test1, y_test1, reference=lgb_train)
        lgb_valid2 = lgb.Dataset(X_test2, y_test2, reference=lgb_train)

        iter_valid1_l1 = 3
        iter_valid1_l2 = 14
        iter_valid2_l1 = 2
        iter_valid2_l2 = 15
        self.assertEqual(len(set([iter_valid1_l1, iter_valid1_l2, iter_valid2_l1, iter_valid2_l2])), 4)
        iter_min_l1 = min([iter_valid1_l1, iter_valid2_l1])
        iter_min_l2 = min([iter_valid1_l2, iter_valid2_l2])
        iter_min = min([iter_min_l1, iter_min_l2])
        iter_min_valid1 = min([iter_valid1_l1, iter_valid1_l2])

        iter_cv_l1 = 3
        iter_cv_l2 = 17
        self.assertEqual(len(set([iter_cv_l1, iter_cv_l2])), 2)
        iter_cv_min = min([iter_cv_l1, iter_cv_l2])

        # test for lgb.train
        metrics_combination_train_regression(lgb_valid1, [], iter_valid1_l2, False)
        metrics_combination_train_regression(lgb_valid1, [], iter_valid1_l2, True)
        metrics_combination_train_regression(lgb_valid1, None, iter_valid1_l2, False)
        metrics_combination_train_regression(lgb_valid1, None, iter_valid1_l2, True)
        metrics_combination_train_regression(lgb_valid1, 'l2', iter_valid1_l2, True)
        metrics_combination_train_regression(lgb_valid1, 'l1', iter_valid1_l1, True)
        metrics_combination_train_regression(lgb_valid1, ['l2', 'l1'], iter_valid1_l2, True)
        metrics_combination_train_regression(lgb_valid1, ['l1', 'l2'], iter_valid1_l1, True)
        metrics_combination_train_regression(lgb_valid1, ['l2', 'l1'], iter_min_valid1, False)
        metrics_combination_train_regression(lgb_valid1, ['l1', 'l2'], iter_min_valid1, False)

        # test feval for lgb.train
        metrics_combination_train_regression(lgb_valid1, 'None', 1, False,
                                             feval=lambda preds, train_data: [decreasing_metric(preds, train_data),
                                                                              constant_metric(preds, train_data)])
        metrics_combination_train_regression(lgb_valid1, 'None', 25, True,
                                             feval=lambda preds, train_data: [decreasing_metric(preds, train_data),
                                                                              constant_metric(preds, train_data)])
        metrics_combination_train_regression(lgb_valid1, 'None', 1, True,
                                             feval=lambda preds, train_data: [constant_metric(preds, train_data),
                                                                              decreasing_metric(preds, train_data)])

        # test with two valid data for lgb.train
        metrics_combination_train_regression([lgb_valid1, lgb_valid2], ['l2', 'l1'], iter_min_l2, True)
        metrics_combination_train_regression([lgb_valid2, lgb_valid1], ['l2', 'l1'], iter_min_l2, True)
        metrics_combination_train_regression([lgb_valid1, lgb_valid2], ['l1', 'l2'], iter_min_l1, True)
        metrics_combination_train_regression([lgb_valid2, lgb_valid1], ['l1', 'l2'], iter_min_l1, True)

        # test for lgb.cv
        metrics_combination_cv_regression(None, iter_cv_l2, True, False)
        metrics_combination_cv_regression('l2', iter_cv_l2, True, False)
        metrics_combination_cv_regression('l1', iter_cv_l1, True, False)
        metrics_combination_cv_regression(['l2', 'l1'], iter_cv_l2, True, False)
        metrics_combination_cv_regression(['l1', 'l2'], iter_cv_l1, True, False)
        metrics_combination_cv_regression(['l2', 'l1'], iter_cv_min, False, False)
        metrics_combination_cv_regression(['l1', 'l2'], iter_cv_min, False, False)
        metrics_combination_cv_regression(None, iter_cv_l2, True, True)
        metrics_combination_cv_regression('l2', iter_cv_l2, True, True)
        metrics_combination_cv_regression('l1', iter_cv_l1, True, True)
        metrics_combination_cv_regression(['l2', 'l1'], iter_cv_l2, True, True)
        metrics_combination_cv_regression(['l1', 'l2'], iter_cv_l1, True, True)
        metrics_combination_cv_regression(['l2', 'l1'], iter_cv_min, False, True)
        metrics_combination_cv_regression(['l1', 'l2'], iter_cv_min, False, True)

        # test feval for lgb.cv
        metrics_combination_cv_regression('None', 1, False, False,
                                          feval=lambda preds, train_data: [decreasing_metric(preds, train_data),
                                                                           constant_metric(preds, train_data)])
        metrics_combination_cv_regression('None', 25, True, False,
                                          feval=lambda preds, train_data: [decreasing_metric(preds, train_data),
                                                                           constant_metric(preds, train_data)])
        metrics_combination_cv_regression('None', 1, True, False,
                                          feval=lambda preds, train_data: [constant_metric(preds, train_data),
                                                                           decreasing_metric(preds, train_data)])

    def test_node_level_subcol(self):
        X, y = load_breast_cancer(True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'feature_fraction_bynode': 0.8,
            'feature_fraction': 1.0,
            'verbose': -1
        }
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
        evals_result = {}
        gbm = lgb.train(params, lgb_train,
                        num_boost_round=25,
                        valid_sets=lgb_eval,
                        verbose_eval=False,
                        evals_result=evals_result)
        ret = log_loss(y_test, gbm.predict(X_test))
        self.assertLess(ret, 0.13)
        self.assertAlmostEqual(evals_result['valid_0']['binary_logloss'][-1], ret, places=5)
        params['feature_fraction'] = 0.5
        gbm2 = lgb.train(params, lgb_train,
                         num_boost_round=25)
        ret2 = log_loss(y_test, gbm2.predict(X_test))
        self.assertNotEqual(ret, ret2)

    def test_forced_bins(self):
        x = np.zeros((100, 2))
        x[:, 0] = np.arange(0, 1, 0.01)
        x[:, 1] = -np.arange(0, 1, 0.01)
        y = np.arange(0, 1, 0.01)
        forcedbins_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                           '../../examples/regression/forced_bins.json')
        params = {'objective': 'regression_l1',
                  'max_bin': 5,
                  'forcedbins_filename': forcedbins_filename,
                  'num_leaves': 2,
                  'min_data_in_leaf': 1,
                  'verbose': -1,
                  'seed': 0}
        lgb_x = lgb.Dataset(x, label=y)
        est = lgb.train(params, lgb_x, num_boost_round=100)
        new_x = np.zeros((3, x.shape[1]))
        new_x[:, 0] = [0.31, 0.37, 0.41]
        new_x[:, 1] = [0, 0, 0]
        predicted = est.predict(new_x)
        self.assertEqual(len(np.unique(predicted)), 3)
        new_x[:, 0] = [0, 0, 0]
        new_x[:, 1] = [-0.9, -0.6, -0.3]
        predicted = est.predict(new_x)
        self.assertEqual(len(np.unique(predicted)), 1)
        params['forcedbins_filename'] = ''
        lgb_x = lgb.Dataset(x, label=y)
        est = lgb.train(params, lgb_x, num_boost_round=100)
        predicted = est.predict(new_x)
        self.assertEqual(len(np.unique(predicted)), 3)
        params['forcedbins_filename'] = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                     '../../examples/regression/forced_bins2.json')
        params['max_bin'] = 11
        lgb_x = lgb.Dataset(x[:, :1], label=y)
        est = lgb.train(params, lgb_x, num_boost_round=100)
        predicted = est.predict(x[1:, :1])
        vals, counts = np.unique(predicted, return_counts=True)
        self.assertGreaterEqual(min(counts), 9)
        self.assertLessEqual(max(counts), 11)

    def test_binning_same_sign(self):
        # test that binning works properly for features with only positive or only negative values
        x = np.zeros((99, 2))
        x[:, 0] = np.arange(0.01, 1, 0.01)
        x[:, 1] = -np.arange(0.01, 1, 0.01)
        y = np.arange(0.01, 1, 0.01)
        params = {'objective': 'regression_l1',
                  'max_bin': 5,
                  'num_leaves': 2,
                  'min_data_in_leaf': 1,
                  'verbose': -1,
                  'seed': 0}
        lgb_x = lgb.Dataset(x, label=y)
        est = lgb.train(params, lgb_x, num_boost_round=100)
        new_x = np.zeros((3, 2))
        new_x[:, 0] = [-1, 0, 1]
        predicted = est.predict(new_x)
        self.assertAlmostEqual(predicted[0], predicted[1])
        self.assertNotAlmostEqual(predicted[1], predicted[2])
        new_x = np.zeros((3, 2))
        new_x[:, 1] = [-1, 0, 1]
        predicted = est.predict(new_x)
        self.assertNotAlmostEqual(predicted[0], predicted[1])
        self.assertAlmostEqual(predicted[1], predicted[2])
