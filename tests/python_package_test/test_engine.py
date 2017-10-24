# coding: utf-8
# pylint: skip-file
import copy
import math
import os
import unittest

import lightgbm as lgb
import random
import numpy as np
from sklearn.datasets import (load_boston, load_breast_cancer, load_digits,
                              load_iris, load_svmlight_file)
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
        np.testing.assert_almost_equal(pred, y)

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
        np.testing.assert_almost_equal(pred, y)

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
        np.testing.assert_almost_equal(pred, y)

    def test_categorical_handle2(self):
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
        np.testing.assert_almost_equal(pred, y)

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
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train, params=params)
        evals_result = {}
        gbm = lgb.train(params, lgb_train,
                        num_boost_round=50,
                        valid_sets=lgb_eval,
                        verbose_eval=False,
                        evals_result=evals_result)

        pred_parameter = {"pred_early_stop": True, "pred_early_stop_freq": 5, "pred_early_stop_margin": 1.5}
        ret = multi_logloss(y_test, gbm.predict(X_test, pred_parameter=pred_parameter))
        self.assertLess(ret, 0.8)
        self.assertGreater(ret, 0.5)  # loss will be higher than when evaluating the full model

        pred_parameter = {"pred_early_stop": True, "pred_early_stop_freq": 5, "pred_early_stop_margin": 5.5}
        ret = multi_logloss(y_test, gbm.predict(X_test, pred_parameter=pred_parameter))
        self.assertLess(ret, 0.2)

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
        self.assertEqual(gbm.best_iteration, 0)
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
        lgb.cv(params_with_metric, lgb_train, num_boost_round=10, nfold=3, stratified=False, shuffle=False,
               metrics='l1', verbose_eval=False)
        # shuffle = True, callbacks
        lgb.cv(params, lgb_train, num_boost_round=10, nfold=3, stratified=False, shuffle=True,
               metrics='l1', verbose_eval=False,
               callbacks=[lgb.reset_parameter(learning_rate=lambda i: 0.1 - 0.001 * i)])
        # self defined folds
        tss = TimeSeriesSplit(3)
        folds = tss.split(X_train)
        lgb.cv(params_with_metric, lgb_train, num_boost_round=10, folds=folds, stratified=False, verbose_eval=False)
        # lambdarank
        X_train, y_train = load_svmlight_file(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../examples/lambdarank/rank.train'))
        q_train = np.loadtxt(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../examples/lambdarank/rank.train.query'))
        params_lambdarank = {'objective': 'lambdarank', 'verbose': -1}
        lgb_train = lgb.Dataset(X_train, y_train, group=q_train)
        lgb.cv(params_lambdarank, lgb_train, num_boost_round=10, nfold=3, stratified=False, metrics='l2', verbose_eval=False)

    def test_feature_name(self):
        X, y = load_boston(True)
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.1, random_state=42)
        params = {'verbose': -1}
        lgb_train = lgb.Dataset(X_train, y_train)
        feature_names = ['f_' + str(i) for i in range(13)]
        gbm = lgb.train(params, lgb_train, num_boost_round=5, feature_name=feature_names)
        self.assertListEqual(feature_names, gbm.feature_name())
        # test feature_names with whitespaces
        feature_names_with_space = ['f ' + str(i) for i in range(13)]
        gbm = lgb.train(params, lgb_train, num_boost_round=5, feature_name=feature_names_with_space)
        self.assertListEqual(feature_names, gbm.feature_name())

    def test_save_load_copy_pickle(self):
        def test_template(init_model=None, return_model=False):
            X, y = load_boston(True)
            params = {
                'objective': 'regression',
                'metric': 'l2',
                'verbose': -1
            }
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
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
        gbm3.save_model('categorical.model')
        gbm4 = lgb.Booster(model_file='categorical.model')
        pred4 = list(gbm4.predict(X_test))
        np.testing.assert_almost_equal(pred0, pred1)
        np.testing.assert_almost_equal(pred0, pred2)
        np.testing.assert_almost_equal(pred0, pred3)
        np.testing.assert_almost_equal(pred0, pred4)

    def test_reference_chain(self):
        X = np.random.normal(size=(100, 2))
        y = np.random.normal(size=100)
        tmp_dat = lgb.Dataset(X, y)
        # take subsets and train
        tmp_dat_train = tmp_dat.subset(np.arange(80))
        tmp_dat_val = tmp_dat.subset(np.arange(80, 100)).subset(np.arange(18))
        params = {'objective': 'regression_l2', 'metric': 'rmse'}
        gbm = lgb.train(params, tmp_dat_train, num_boost_round=20, valid_sets=[tmp_dat_train, tmp_dat_val])

    def test_contribs(self):
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

        self.assertLess(np.linalg.norm(gbm.predict(X_test, raw_score=True) - np.sum(gbm.predict(X_test, pred_contrib=True), axis=1)), 1e-4)
