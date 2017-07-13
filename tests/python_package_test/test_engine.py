# coding: utf-8
# pylint: skip-file
import copy
import math
import os
import unittest

import lightgbm as lgb
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
        self.assertLess(ret, 0.15)
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

    def test_regreesion(self):
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

    def test_continue_train_and_dump_model(self):
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
        # test dump model
        self.assertIn('tree_info', gbm.dump_model())
        self.assertIsInstance(gbm.feature_importance(), np.ndarray)
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
        lgb.cv(params_with_metric, lgb_train, num_boost_round=10, nfold=3, shuffle=False,
               metrics='l1', verbose_eval=False)
        # shuffle = True, callbacks
        lgb.cv(params, lgb_train, num_boost_round=10, nfold=3, shuffle=True,
               metrics='l1', verbose_eval=False,
               callbacks=[lgb.reset_parameter(learning_rate=lambda i: 0.1 - 0.001 * i)])
        # self defined folds
        tss = TimeSeriesSplit(3)
        folds = tss.split(X_train)
        lgb.cv(params_with_metric, lgb_train, num_boost_round=10, folds=folds, verbose_eval=False)
        # lambdarank
        X_train, y_train = load_svmlight_file(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../examples/lambdarank/rank.train'))
        q_train = np.loadtxt(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../examples/lambdarank/rank.train.query'))
        params_lambdarank = {'objective': 'lambdarank', 'verbose': -1}
        lgb_train = lgb.Dataset(X_train, y_train, group=q_train)
        lgb.cv(params_lambdarank, lgb_train, num_boost_round=10, nfold=3, metrics='l2', verbose_eval=False)

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
        lgb_train = lgb.Dataset(X, y)
        gbm3.save_model('categorical.model')
        gbm4 = lgb.Booster(model_file='categorical.model')
        pred4 = list(gbm4.predict(X_test))
        np.testing.assert_almost_equal(pred0, pred1)
        np.testing.assert_almost_equal(pred0, pred2)
        np.testing.assert_almost_equal(pred0, pred3)
        np.testing.assert_almost_equal(pred0, pred4)
