# coding: utf-8
import copy
import itertools
import math
import os
import pickle
import platform
import random

import numpy as np
import psutil
import pytest
from scipy.sparse import csr_matrix, isspmatrix_csc, isspmatrix_csr
from sklearn.datasets import load_svmlight_file, make_multilabel_classification
from sklearn.metrics import average_precision_score, log_loss, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import GroupKFold, TimeSeriesSplit, train_test_split

import lightgbm as lgb

from .utils import load_boston, load_breast_cancer, load_digits, load_iris

decreasing_generator = itertools.count(0, -1)


def dummy_obj(preds, train_data):
    return np.ones(preds.shape), np.ones(preds.shape)


def multi_logloss(y_true, y_pred):
    return np.mean([-math.log(y_pred[i][y]) for i, y in enumerate(y_true)])


def top_k_error(y_true, y_pred, k):
    if k == y_pred.shape[1]:
        return 0
    max_rest = np.max(-np.partition(-y_pred, k)[:, k:], axis=1)
    return 1 - np.mean((y_pred[np.arange(len(y_true)), y_true] > max_rest))


def constant_metric(preds, train_data):
    return ('error', 0.0, False)


def decreasing_metric(preds, train_data):
    return ('decreasing_metric', next(decreasing_generator), False)


def categorize(continuous_x):
    return np.digitize(continuous_x, bins=np.arange(0, 1, 0.01))


def test_binary():
    X, y = load_breast_cancer(return_X_y=True)
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
    assert ret < 0.14
    assert len(evals_result['valid_0']['binary_logloss']) == 50
    assert evals_result['valid_0']['binary_logloss'][-1] == pytest.approx(ret)


def test_rf():
    X, y = load_breast_cancer(return_X_y=True)
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
    assert ret < 0.19
    assert evals_result['valid_0']['binary_logloss'][-1] == pytest.approx(ret)


def test_regression():
    X, y = load_boston(return_X_y=True)
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
    assert ret < 7
    assert evals_result['valid_0']['l2'][-1] == pytest.approx(ret)


def test_missing_value_handle():
    X_train = np.zeros((100, 1))
    y_train = np.zeros(100)
    trues = random.sample(range(100), 20)
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
                    verbose_eval=False,
                    evals_result=evals_result)
    ret = mean_squared_error(y_train, gbm.predict(X_train))
    assert ret < 0.005
    assert evals_result['valid_0']['l2'][-1] == pytest.approx(ret)


def test_missing_value_handle_more_na():
    X_train = np.ones((100, 1))
    y_train = np.ones(100)
    trues = random.sample(range(100), 80)
    for idx in trues:
        X_train[idx, 0] = np.nan
        y_train[idx] = 0
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
                    verbose_eval=False,
                    evals_result=evals_result)
    ret = mean_squared_error(y_train, gbm.predict(X_train))
    assert ret < 0.005
    assert evals_result['valid_0']['l2'][-1] == pytest.approx(ret)


def test_missing_value_handle_na():
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
                    verbose_eval=False,
                    evals_result=evals_result)
    pred = gbm.predict(X_train)
    np.testing.assert_allclose(pred, y)
    ret = roc_auc_score(y_train, pred)
    assert ret > 0.999
    assert evals_result['valid_0']['auc'][-1] == pytest.approx(ret)


def test_missing_value_handle_zero():
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
                    verbose_eval=False,
                    evals_result=evals_result)
    pred = gbm.predict(X_train)
    np.testing.assert_allclose(pred, y)
    ret = roc_auc_score(y_train, pred)
    assert ret > 0.999
    assert evals_result['valid_0']['auc'][-1] == pytest.approx(ret)


def test_missing_value_handle_none():
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
                    verbose_eval=False,
                    evals_result=evals_result)
    pred = gbm.predict(X_train)
    assert pred[0] == pytest.approx(pred[1])
    assert pred[-1] == pytest.approx(pred[0])
    ret = roc_auc_score(y_train, pred)
    assert ret > 0.83
    assert evals_result['valid_0']['auc'][-1] == pytest.approx(ret)


def test_categorical_handle():
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
                    verbose_eval=False,
                    evals_result=evals_result)
    pred = gbm.predict(X_train)
    np.testing.assert_allclose(pred, y)
    ret = roc_auc_score(y_train, pred)
    assert ret > 0.999
    assert evals_result['valid_0']['auc'][-1] == pytest.approx(ret)


def test_categorical_handle_na():
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
                    verbose_eval=False,
                    evals_result=evals_result)
    pred = gbm.predict(X_train)
    np.testing.assert_allclose(pred, y)
    ret = roc_auc_score(y_train, pred)
    assert ret > 0.999
    assert evals_result['valid_0']['auc'][-1] == pytest.approx(ret)


def test_categorical_non_zero_inputs():
    x = [1, 1, 1, 1, 1, 1, 2, 2]
    y = [1, 1, 1, 1, 1, 1, 0, 0]

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
                    verbose_eval=False,
                    evals_result=evals_result)
    pred = gbm.predict(X_train)
    np.testing.assert_allclose(pred, y)
    ret = roc_auc_score(y_train, pred)
    assert ret > 0.999
    assert evals_result['valid_0']['auc'][-1] == pytest.approx(ret)


def test_multiclass():
    X, y = load_digits(n_class=10, return_X_y=True)
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
    assert ret < 0.16
    assert evals_result['valid_0']['multi_logloss'][-1] == pytest.approx(ret)


def test_multiclass_rf():
    X, y = load_digits(n_class=10, return_X_y=True)
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
        'verbose': -1,
        'gpu_use_dp': True
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
    assert ret < 0.23
    assert evals_result['valid_0']['multi_logloss'][-1] == pytest.approx(ret)


def test_multiclass_prediction_early_stopping():
    X, y = load_digits(n_class=10, return_X_y=True)
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
    assert ret < 0.8
    assert ret > 0.6  # loss will be higher than when evaluating the full model

    pred_parameter = {"pred_early_stop": True,
                      "pred_early_stop_freq": 5,
                      "pred_early_stop_margin": 5.5}
    ret = multi_logloss(y_test, gbm.predict(X_test, **pred_parameter))
    assert ret < 0.2


def test_multi_class_error():
    X, y = load_digits(n_class=10, return_X_y=True)
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
    assert results['training']['multi_error'][-1] == pytest.approx(err)
    # check against independent calculation for k = 2
    results = {}
    est = lgb.train(dict(params, multi_error_top_k=2), lgb_data, num_boost_round=10,
                    valid_sets=[lgb_data], evals_result=results, verbose_eval=False)
    predict_2 = est.predict(X)
    err = top_k_error(y, predict_2, 2)
    assert results['training']['multi_error@2'][-1] == pytest.approx(err)
    # check against independent calculation for k = 10
    results = {}
    est = lgb.train(dict(params, multi_error_top_k=10), lgb_data, num_boost_round=10,
                    valid_sets=[lgb_data], evals_result=results, verbose_eval=False)
    predict_3 = est.predict(X)
    err = top_k_error(y, predict_3, 10)
    assert results['training']['multi_error@10'][-1] == pytest.approx(err)
    # check cases where predictions are equal
    X = np.array([[0, 0], [0, 0]])
    y = np.array([0, 1])
    lgb_data = lgb.Dataset(X, label=y)
    params['num_classes'] = 2
    results = {}
    lgb.train(params, lgb_data, num_boost_round=10,
              valid_sets=[lgb_data], evals_result=results, verbose_eval=False)
    assert results['training']['multi_error'][-1] == pytest.approx(1)
    results = {}
    lgb.train(dict(params, multi_error_top_k=2), lgb_data, num_boost_round=10,
              valid_sets=[lgb_data], evals_result=results, verbose_eval=False)
    assert results['training']['multi_error@2'][-1] == pytest.approx(0)


def test_auc_mu():
    # should give same result as binary auc for 2 classes
    X, y = load_digits(n_class=10, return_X_y=True)
    y_new = np.zeros((len(y)))
    y_new[y != 0] = 1
    lgb_X = lgb.Dataset(X, label=y_new)
    params = {'objective': 'multiclass',
              'metric': 'auc_mu',
              'verbose': -1,
              'num_classes': 2,
              'seed': 0}
    results_auc_mu = {}
    lgb.train(params, lgb_X, num_boost_round=10, valid_sets=[lgb_X], evals_result=results_auc_mu)
    params = {'objective': 'binary',
              'metric': 'auc',
              'verbose': -1,
              'seed': 0}
    results_auc = {}
    lgb.train(params, lgb_X, num_boost_round=10, valid_sets=[lgb_X], evals_result=results_auc)
    np.testing.assert_allclose(results_auc_mu['training']['auc_mu'], results_auc['training']['auc'])
    # test the case where all predictions are equal
    lgb_X = lgb.Dataset(X[:10], label=y_new[:10])
    params = {'objective': 'multiclass',
              'metric': 'auc_mu',
              'verbose': -1,
              'num_classes': 2,
              'min_data_in_leaf': 20,
              'seed': 0}
    results_auc_mu = {}
    lgb.train(params, lgb_X, num_boost_round=10, valid_sets=[lgb_X], evals_result=results_auc_mu)
    assert results_auc_mu['training']['auc_mu'][-1] == pytest.approx(0.5)
    # test that weighted data gives different auc_mu
    lgb_X = lgb.Dataset(X, label=y)
    lgb_X_weighted = lgb.Dataset(X, label=y, weight=np.abs(np.random.normal(size=y.shape)))
    results_unweighted = {}
    results_weighted = {}
    params = dict(params, num_classes=10, num_leaves=5)
    lgb.train(params, lgb_X, num_boost_round=10, valid_sets=[lgb_X], evals_result=results_unweighted)
    lgb.train(params, lgb_X_weighted, num_boost_round=10, valid_sets=[lgb_X_weighted],
              evals_result=results_weighted)
    assert results_weighted['training']['auc_mu'][-1] < 1
    assert results_unweighted['training']['auc_mu'][-1] != results_weighted['training']['auc_mu'][-1]
    # test that equal data weights give same auc_mu as unweighted data
    lgb_X_weighted = lgb.Dataset(X, label=y, weight=np.ones(y.shape) * 0.5)
    lgb.train(params, lgb_X_weighted, num_boost_round=10, valid_sets=[lgb_X_weighted],
              evals_result=results_weighted)
    assert results_unweighted['training']['auc_mu'][-1] == pytest.approx(
        results_weighted['training']['auc_mu'][-1], abs=1e-5)
    # should give 1 when accuracy = 1
    X = X[:10, :]
    y = y[:10]
    lgb_X = lgb.Dataset(X, label=y)
    params = {'objective': 'multiclass',
              'metric': 'auc_mu',
              'num_classes': 10,
              'min_data_in_leaf': 1,
              'verbose': -1}
    results = {}
    lgb.train(params, lgb_X, num_boost_round=100, valid_sets=[lgb_X], evals_result=results)
    assert results['training']['auc_mu'][-1] == pytest.approx(1)
    # test loading class weights
    Xy = np.loadtxt(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 '../../examples/multiclass_classification/multiclass.train'))
    y = Xy[:, 0]
    X = Xy[:, 1:]
    lgb_X = lgb.Dataset(X, label=y)
    params = {'objective': 'multiclass',
              'metric': 'auc_mu',
              'auc_mu_weights': [0, 2, 2, 2, 2, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
              'num_classes': 5,
              'verbose': -1,
              'seed': 0}
    results_weight = {}
    lgb.train(params, lgb_X, num_boost_round=5, valid_sets=[lgb_X], evals_result=results_weight)
    params['auc_mu_weights'] = []
    results_no_weight = {}
    lgb.train(params, lgb_X, num_boost_round=5, valid_sets=[lgb_X], evals_result=results_no_weight)
    assert results_weight['training']['auc_mu'][-1] != results_no_weight['training']['auc_mu'][-1]


def test_early_stopping():
    X, y = load_breast_cancer(return_X_y=True)
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
    assert gbm.best_iteration == 10
    assert valid_set_name in gbm.best_score
    assert 'binary_logloss' in gbm.best_score[valid_set_name]
    # early stopping occurs
    gbm = lgb.train(params, lgb_train,
                    num_boost_round=40,
                    valid_sets=lgb_eval,
                    valid_names=valid_set_name,
                    verbose_eval=False,
                    early_stopping_rounds=5)
    assert gbm.best_iteration <= 39
    assert valid_set_name in gbm.best_score
    assert 'binary_logloss' in gbm.best_score[valid_set_name]


def test_continue_train():
    X, y = load_boston(return_X_y=True)
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
                    feval=(lambda p, d: ('custom_mae', mean_absolute_error(p, d.get_label()), False)),
                    evals_result=evals_result,
                    init_model='model.txt')
    ret = mean_absolute_error(y_test, gbm.predict(X_test))
    assert ret < 2.0
    assert evals_result['valid_0']['l1'][-1] == pytest.approx(ret)
    np.testing.assert_allclose(evals_result['valid_0']['l1'], evals_result['valid_0']['custom_mae'])
    os.remove(model_name)


def test_continue_train_reused_dataset():
    X, y = load_boston(return_X_y=True)
    params = {
        'objective': 'regression',
        'verbose': -1
    }
    lgb_train = lgb.Dataset(X, y, free_raw_data=False)
    init_gbm = lgb.train(params, lgb_train, num_boost_round=5)
    init_gbm_2 = lgb.train(params, lgb_train, num_boost_round=5, init_model=init_gbm)
    init_gbm_3 = lgb.train(params, lgb_train, num_boost_round=5, init_model=init_gbm_2)
    gbm = lgb.train(params, lgb_train, num_boost_round=5, init_model=init_gbm_3)
    assert gbm.current_iteration() == 20


def test_continue_train_dart():
    X, y = load_boston(return_X_y=True)
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
    assert ret < 2.0
    assert evals_result['valid_0']['l1'][-1] == pytest.approx(ret)


def test_continue_train_multiclass():
    X, y = load_iris(return_X_y=True)
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
    assert ret < 0.1
    assert evals_result['valid_0']['multi_logloss'][-1] == pytest.approx(ret)


def test_cv():
    X_train, y_train = load_boston(return_X_y=True)
    params = {'verbose': -1}
    lgb_train = lgb.Dataset(X_train, y_train)
    # shuffle = False, override metric in params
    params_with_metric = {'metric': 'l2', 'verbose': -1}
    cv_res = lgb.cv(params_with_metric, lgb_train, num_boost_round=10,
                    nfold=3, stratified=False, shuffle=False,
                    metrics='l1', verbose_eval=False)
    assert 'l1-mean' in cv_res
    assert 'l2-mean' not in cv_res
    assert len(cv_res['l1-mean']) == 10
    # shuffle = True, callbacks
    cv_res = lgb.cv(params, lgb_train, num_boost_round=10, nfold=3, stratified=False, shuffle=True,
                    metrics='l1', verbose_eval=False,
                    callbacks=[lgb.reset_parameter(learning_rate=lambda i: 0.1 - 0.001 * i)])
    assert 'l1-mean' in cv_res
    assert len(cv_res['l1-mean']) == 10
    # enable display training loss
    cv_res = lgb.cv(params_with_metric, lgb_train, num_boost_round=10,
                    nfold=3, stratified=False, shuffle=False,
                    metrics='l1', verbose_eval=False, eval_train_metric=True)
    assert 'train l1-mean' in cv_res
    assert 'valid l1-mean' in cv_res
    assert 'train l2-mean' not in cv_res
    assert 'valid l2-mean' not in cv_res
    assert len(cv_res['train l1-mean']) == 10
    assert len(cv_res['valid l1-mean']) == 10
    # self defined folds
    tss = TimeSeriesSplit(3)
    folds = tss.split(X_train)
    cv_res_gen = lgb.cv(params_with_metric, lgb_train, num_boost_round=10, folds=folds,
                        verbose_eval=False)
    cv_res_obj = lgb.cv(params_with_metric, lgb_train, num_boost_round=10, folds=tss,
                        verbose_eval=False)
    np.testing.assert_allclose(cv_res_gen['l2-mean'], cv_res_obj['l2-mean'])
    # LambdaRank
    X_train, y_train = load_svmlight_file(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                       '../../examples/lambdarank/rank.train'))
    q_train = np.loadtxt(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                      '../../examples/lambdarank/rank.train.query'))
    params_lambdarank = {'objective': 'lambdarank', 'verbose': -1, 'eval_at': 3}
    lgb_train = lgb.Dataset(X_train, y_train, group=q_train)
    # ... with l2 metric
    cv_res_lambda = lgb.cv(params_lambdarank, lgb_train, num_boost_round=10, nfold=3,
                           metrics='l2', verbose_eval=False)
    assert len(cv_res_lambda) == 2
    assert not np.isnan(cv_res_lambda['l2-mean']).any()
    # ... with NDCG (default) metric
    cv_res_lambda = lgb.cv(params_lambdarank, lgb_train, num_boost_round=10, nfold=3,
                           verbose_eval=False)
    assert len(cv_res_lambda) == 2
    assert not np.isnan(cv_res_lambda['ndcg@3-mean']).any()
    # self defined folds with lambdarank
    cv_res_lambda_obj = lgb.cv(params_lambdarank, lgb_train, num_boost_round=10,
                               folds=GroupKFold(n_splits=3),
                               verbose_eval=False)
    np.testing.assert_allclose(cv_res_lambda['ndcg@3-mean'], cv_res_lambda_obj['ndcg@3-mean'])


def test_cvbooster():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbose': -1,
    }
    lgb_train = lgb.Dataset(X_train, y_train)
    # with early stopping
    cv_res = lgb.cv(params, lgb_train,
                    num_boost_round=25,
                    early_stopping_rounds=5,
                    verbose_eval=False,
                    nfold=3,
                    return_cvbooster=True)
    assert 'cvbooster' in cv_res
    cvb = cv_res['cvbooster']
    assert isinstance(cvb, lgb.CVBooster)
    assert isinstance(cvb.boosters, list)
    assert len(cvb.boosters) == 3
    assert all(isinstance(bst, lgb.Booster) for bst in cvb.boosters)
    assert cvb.best_iteration > 0
    # predict by each fold booster
    preds = cvb.predict(X_test, num_iteration=cvb.best_iteration)
    assert isinstance(preds, list)
    assert len(preds) == 3
    # fold averaging
    avg_pred = np.mean(preds, axis=0)
    ret = log_loss(y_test, avg_pred)
    assert ret < 0.13
    # without early stopping
    cv_res = lgb.cv(params, lgb_train,
                    num_boost_round=20,
                    verbose_eval=False,
                    nfold=3,
                    return_cvbooster=True)
    cvb = cv_res['cvbooster']
    assert cvb.best_iteration == -1
    preds = cvb.predict(X_test)
    avg_pred = np.mean(preds, axis=0)
    ret = log_loss(y_test, avg_pred)
    assert ret < 0.15


def test_feature_name():
    X_train, y_train = load_boston(return_X_y=True)
    params = {'verbose': -1}
    lgb_train = lgb.Dataset(X_train, y_train)
    feature_names = ['f_' + str(i) for i in range(X_train.shape[-1])]
    gbm = lgb.train(params, lgb_train, num_boost_round=5, feature_name=feature_names)
    assert feature_names == gbm.feature_name()
    # test feature_names with whitespaces
    feature_names_with_space = ['f ' + str(i) for i in range(X_train.shape[-1])]
    gbm = lgb.train(params, lgb_train, num_boost_round=5, feature_name=feature_names_with_space)
    assert feature_names == gbm.feature_name()


def test_feature_name_with_non_ascii():
    X_train = np.random.normal(size=(100, 4))
    y_train = np.random.random(100)
    # This has non-ascii strings.
    feature_names = [u'F_零', u'F_一', u'F_二', u'F_三']
    params = {'verbose': -1}
    lgb_train = lgb.Dataset(X_train, y_train)

    gbm = lgb.train(params, lgb_train, num_boost_round=5, feature_name=feature_names)
    assert feature_names == gbm.feature_name()
    gbm.save_model('lgb.model')

    gbm2 = lgb.Booster(model_file='lgb.model')
    assert feature_names == gbm2.feature_name()


def test_save_load_copy_pickle():
    def train_and_predict(init_model=None, return_model=False):
        X, y = load_boston(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        params = {
            'objective': 'regression',
            'metric': 'l2',
            'verbose': -1
        }
        lgb_train = lgb.Dataset(X_train, y_train)
        gbm_template = lgb.train(params, lgb_train, num_boost_round=10, init_model=init_model)
        return gbm_template if return_model else mean_squared_error(y_test, gbm_template.predict(X_test))

    gbm = train_and_predict(return_model=True)
    ret_origin = train_and_predict(init_model=gbm)
    other_ret = []
    gbm.save_model('lgb.model')
    with open('lgb.model') as f:  # check all params are logged into model file correctly
        assert f.read().find("[num_iterations: 10]") != -1
    other_ret.append(train_and_predict(init_model='lgb.model'))
    gbm_load = lgb.Booster(model_file='lgb.model')
    other_ret.append(train_and_predict(init_model=gbm_load))
    other_ret.append(train_and_predict(init_model=copy.copy(gbm)))
    other_ret.append(train_and_predict(init_model=copy.deepcopy(gbm)))
    with open('lgb.pkl', 'wb') as f:
        pickle.dump(gbm, f)
    with open('lgb.pkl', 'rb') as f:
        gbm_pickle = pickle.load(f)
    other_ret.append(train_and_predict(init_model=gbm_pickle))
    gbm_pickles = pickle.loads(pickle.dumps(gbm))
    other_ret.append(train_and_predict(init_model=gbm_pickles))
    for ret in other_ret:
        assert ret_origin == pytest.approx(ret)


def test_pandas_categorical():
    pd = pytest.importorskip("pandas")
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
                           "E": pd.Categorical(np.random.permutation(['z', 'y'] * 30),
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
    assert lgb_train.categorical_feature == 'auto'
    lgb_train = lgb.Dataset(X, pd.DataFrame(y))  # also test that label can be one-column pd.DataFrame
    gbm1 = lgb.train(params, lgb_train, num_boost_round=10, categorical_feature=[0])
    pred1 = gbm1.predict(X_test)
    assert lgb_train.categorical_feature == [0]
    lgb_train = lgb.Dataset(X, pd.Series(y))  # also test that label can be pd.Series
    gbm2 = lgb.train(params, lgb_train, num_boost_round=10, categorical_feature=['A'])
    pred2 = gbm2.predict(X_test)
    assert lgb_train.categorical_feature == ['A']
    lgb_train = lgb.Dataset(X, y)
    gbm3 = lgb.train(params, lgb_train, num_boost_round=10, categorical_feature=['A', 'B', 'C', 'D'])
    pred3 = gbm3.predict(X_test)
    assert lgb_train.categorical_feature == ['A', 'B', 'C', 'D']
    gbm3.save_model('categorical.model')
    gbm4 = lgb.Booster(model_file='categorical.model')
    pred4 = gbm4.predict(X_test)
    model_str = gbm4.model_to_string()
    gbm4.model_from_string(model_str, False)
    pred5 = gbm4.predict(X_test)
    gbm5 = lgb.Booster(model_str=model_str)
    pred6 = gbm5.predict(X_test)
    lgb_train = lgb.Dataset(X, y)
    gbm6 = lgb.train(params, lgb_train, num_boost_round=10, categorical_feature=['A', 'B', 'C', 'D', 'E'])
    pred7 = gbm6.predict(X_test)
    assert lgb_train.categorical_feature == ['A', 'B', 'C', 'D', 'E']
    lgb_train = lgb.Dataset(X, y)
    gbm7 = lgb.train(params, lgb_train, num_boost_round=10, categorical_feature=[])
    pred8 = gbm7.predict(X_test)
    assert lgb_train.categorical_feature == []
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(pred0, pred1)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(pred0, pred2)
    np.testing.assert_allclose(pred1, pred2)
    np.testing.assert_allclose(pred0, pred3)
    np.testing.assert_allclose(pred0, pred4)
    np.testing.assert_allclose(pred0, pred5)
    np.testing.assert_allclose(pred0, pred6)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(pred0, pred7)  # ordered cat features aren't treated as cat features by default
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(pred0, pred8)
    assert gbm0.pandas_categorical == cat_values
    assert gbm1.pandas_categorical == cat_values
    assert gbm2.pandas_categorical == cat_values
    assert gbm3.pandas_categorical == cat_values
    assert gbm4.pandas_categorical == cat_values
    assert gbm5.pandas_categorical == cat_values
    assert gbm6.pandas_categorical == cat_values
    assert gbm7.pandas_categorical == cat_values


def test_pandas_sparse():
    pd = pytest.importorskip("pandas")
    try:
        from pandas.arrays import SparseArray
    except ImportError:  # support old versions
        from pandas import SparseArray
    X = pd.DataFrame({"A": SparseArray(np.random.permutation([0, 1, 2] * 100)),
                      "B": SparseArray(np.random.permutation([0.0, 0.1, 0.2, -0.1, 0.2] * 60)),
                      "C": SparseArray(np.random.permutation([True, False] * 150))})
    y = pd.Series(SparseArray(np.random.permutation([0, 1] * 150)))
    X_test = pd.DataFrame({"A": SparseArray(np.random.permutation([0, 2] * 30)),
                           "B": SparseArray(np.random.permutation([0.0, 0.1, 0.2, -0.1] * 15)),
                           "C": SparseArray(np.random.permutation([True, False] * 30))})
    if pd.__version__ >= '0.24.0':
        for dtype in pd.concat([X.dtypes, X_test.dtypes, pd.Series(y.dtypes)]):
            assert pd.api.types.is_sparse(dtype)
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


def test_reference_chain():
    X = np.random.normal(size=(100, 2))
    y = np.random.normal(size=100)
    tmp_dat = lgb.Dataset(X, y)
    # take subsets and train
    tmp_dat_train = tmp_dat.subset(np.arange(80))
    tmp_dat_val = tmp_dat.subset(np.arange(80, 100)).subset(np.arange(18))
    params = {'objective': 'regression_l2', 'metric': 'rmse'}
    evals_result = {}
    lgb.train(params, tmp_dat_train, num_boost_round=20,
              valid_sets=[tmp_dat_train, tmp_dat_val],
              verbose_eval=False, evals_result=evals_result)
    assert len(evals_result['training']['rmse']) == 20
    assert len(evals_result['valid_1']['rmse']) == 20


def test_contribs():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbose': -1,
    }
    lgb_train = lgb.Dataset(X_train, y_train)
    gbm = lgb.train(params, lgb_train, num_boost_round=20)

    assert (np.linalg.norm(gbm.predict(X_test, raw_score=True)
                           - np.sum(gbm.predict(X_test, pred_contrib=True), axis=1)) < 1e-4)


def test_contribs_sparse():
    n_features = 20
    n_samples = 100
    # generate CSR sparse dataset
    X, y = make_multilabel_classification(n_samples=n_samples,
                                          sparse=True,
                                          n_features=n_features,
                                          n_classes=1,
                                          n_labels=2)
    y = y.flatten()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    params = {
        'objective': 'binary',
        'verbose': -1,
    }
    lgb_train = lgb.Dataset(X_train, y_train)
    gbm = lgb.train(params, lgb_train, num_boost_round=20)
    contribs_csr = gbm.predict(X_test, pred_contrib=True)
    assert isspmatrix_csr(contribs_csr)
    # convert data to dense and get back same contribs
    contribs_dense = gbm.predict(X_test.toarray(), pred_contrib=True)
    # validate the values are the same
    if platform.machine() == 'aarch64':
        np.testing.assert_allclose(contribs_csr.toarray(), contribs_dense, rtol=1, atol=1e-12)
    else:
        np.testing.assert_allclose(contribs_csr.toarray(), contribs_dense)
    assert (np.linalg.norm(gbm.predict(X_test, raw_score=True)
                           - np.sum(contribs_dense, axis=1)) < 1e-4)
    # validate using CSC matrix
    X_test_csc = X_test.tocsc()
    contribs_csc = gbm.predict(X_test_csc, pred_contrib=True)
    assert isspmatrix_csc(contribs_csc)
    # validate the values are the same
    if platform.machine() == 'aarch64':
        np.testing.assert_allclose(contribs_csc.toarray(), contribs_dense, rtol=1, atol=1e-12)
    else:
        np.testing.assert_allclose(contribs_csc.toarray(), contribs_dense)


def test_contribs_sparse_multiclass():
    n_features = 20
    n_samples = 100
    n_labels = 4
    # generate CSR sparse dataset
    X, y = make_multilabel_classification(n_samples=n_samples,
                                          sparse=True,
                                          n_features=n_features,
                                          n_classes=1,
                                          n_labels=n_labels)
    y = y.flatten()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    params = {
        'objective': 'multiclass',
        'num_class': n_labels,
        'verbose': -1,
    }
    lgb_train = lgb.Dataset(X_train, y_train)
    gbm = lgb.train(params, lgb_train, num_boost_round=20)
    contribs_csr = gbm.predict(X_test, pred_contrib=True)
    assert isinstance(contribs_csr, list)
    for perclass_contribs_csr in contribs_csr:
        assert isspmatrix_csr(perclass_contribs_csr)
    # convert data to dense and get back same contribs
    contribs_dense = gbm.predict(X_test.toarray(), pred_contrib=True)
    # validate the values are the same
    contribs_csr_array = np.swapaxes(np.array([sparse_array.todense() for sparse_array in contribs_csr]), 0, 1)
    contribs_csr_arr_re = contribs_csr_array.reshape((contribs_csr_array.shape[0],
                                                      contribs_csr_array.shape[1] * contribs_csr_array.shape[2]))
    if platform.machine() == 'aarch64':
        np.testing.assert_allclose(contribs_csr_arr_re, contribs_dense, rtol=1, atol=1e-12)
    else:
        np.testing.assert_allclose(contribs_csr_arr_re, contribs_dense)
    contribs_dense_re = contribs_dense.reshape(contribs_csr_array.shape)
    assert np.linalg.norm(gbm.predict(X_test, raw_score=True) - np.sum(contribs_dense_re, axis=2)) < 1e-4
    # validate using CSC matrix
    X_test_csc = X_test.tocsc()
    contribs_csc = gbm.predict(X_test_csc, pred_contrib=True)
    assert isinstance(contribs_csc, list)
    for perclass_contribs_csc in contribs_csc:
        assert isspmatrix_csc(perclass_contribs_csc)
    # validate the values are the same
    contribs_csc_array = np.swapaxes(np.array([sparse_array.todense() for sparse_array in contribs_csc]), 0, 1)
    contribs_csc_array = contribs_csc_array.reshape((contribs_csc_array.shape[0],
                                                     contribs_csc_array.shape[1] * contribs_csc_array.shape[2]))
    if platform.machine() == 'aarch64':
        np.testing.assert_allclose(contribs_csc_array, contribs_dense, rtol=1, atol=1e-12)
    else:
        np.testing.assert_allclose(contribs_csc_array, contribs_dense)


@pytest.mark.skipif(psutil.virtual_memory().available / 1024 / 1024 / 1024 < 3, reason='not enough RAM')
def test_int32_max_sparse_contribs():
    params = {
        'objective': 'binary'
    }
    train_features = np.random.rand(100, 1000)
    train_targets = [0] * 50 + [1] * 50
    lgb_train = lgb.Dataset(train_features, train_targets)
    gbm = lgb.train(params, lgb_train, num_boost_round=2)
    csr_input_shape = (3000000, 1000)
    test_features = csr_matrix(csr_input_shape)
    for i in range(0, csr_input_shape[0], csr_input_shape[0] // 6):
        for j in range(0, 1000, 100):
            test_features[i, j] = random.random()
    y_pred_csr = gbm.predict(test_features, pred_contrib=True)
    # Note there is an extra column added to the output for the expected value
    csr_output_shape = (csr_input_shape[0], csr_input_shape[1] + 1)
    assert y_pred_csr.shape == csr_output_shape
    y_pred_csc = gbm.predict(test_features.tocsc(), pred_contrib=True)
    # Note output CSC shape should be same as CSR output shape
    assert y_pred_csc.shape == csr_output_shape


def test_sliced_data():
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
    labels = np.append(np.ones(positive_samples, dtype=np.float32),
                       np.zeros(num_samples - positive_samples, dtype=np.float32))
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
    assert np.all(sliced_features == features)
    sliced_pred = train_and_get_predictions(sliced_features, sliced_labels)
    np.testing.assert_allclose(origin_pred, sliced_pred)
    # test sliced CSR
    stacked_csr = csr_matrix(stacked_features)
    sliced_csr = stacked_csr[2:102, 2:7]
    assert np.all(sliced_csr == features)
    sliced_pred = train_and_get_predictions(sliced_csr, sliced_labels)
    np.testing.assert_allclose(origin_pred, sliced_pred)


def test_init_with_subset():
    data = np.random.random((50, 2))
    y = [1] * 25 + [0] * 25
    lgb_train = lgb.Dataset(data, y, free_raw_data=False)
    subset_index_1 = np.random.choice(np.arange(50), 30, replace=False)
    subset_data_1 = lgb_train.subset(subset_index_1)
    subset_index_2 = np.random.choice(np.arange(50), 20, replace=False)
    subset_data_2 = lgb_train.subset(subset_index_2)
    params = {
        'objective': 'binary',
        'verbose': -1
    }
    init_gbm = lgb.train(params=params,
                         train_set=subset_data_1,
                         num_boost_round=10,
                         keep_training_booster=True)
    lgb.train(params=params,
              train_set=subset_data_2,
              num_boost_round=10,
              init_model=init_gbm)
    assert lgb_train.get_data().shape[0] == 50
    assert subset_data_1.get_data().shape[0] == 30
    assert subset_data_2.get_data().shape[0] == 20
    lgb_train.save_binary("lgb_train_data.bin")
    lgb_train_from_file = lgb.Dataset('lgb_train_data.bin', free_raw_data=False)
    subset_data_3 = lgb_train_from_file.subset(subset_index_1)
    subset_data_4 = lgb_train_from_file.subset(subset_index_2)
    init_gbm_2 = lgb.train(params=params,
                           train_set=subset_data_3,
                           num_boost_round=10,
                           keep_training_booster=True)
    with np.testing.assert_raises_regex(lgb.basic.LightGBMError, "Unknown format of training data"):
        lgb.train(params=params,
                  train_set=subset_data_4,
                  num_boost_round=10,
                  init_model=init_gbm_2)
    assert lgb_train_from_file.get_data() == "lgb_train_data.bin"
    assert subset_data_3.get_data() == "lgb_train_data.bin"
    assert subset_data_4.get_data() == "lgb_train_data.bin"


def generate_trainset_for_monotone_constraints_tests(x3_to_category=True):
    number_of_dpoints = 3000
    x1_positively_correlated_with_y = np.random.random(size=number_of_dpoints)
    x2_negatively_correlated_with_y = np.random.random(size=number_of_dpoints)
    x3_negatively_correlated_with_y = np.random.random(size=number_of_dpoints)
    x = np.column_stack(
        (x1_positively_correlated_with_y,
            x2_negatively_correlated_with_y,
            categorize(x3_negatively_correlated_with_y) if x3_to_category else x3_negatively_correlated_with_y))

    zs = np.random.normal(loc=0.0, scale=0.01, size=number_of_dpoints)
    scales = 10. * (np.random.random(6) + 0.5)
    y = (scales[0] * x1_positively_correlated_with_y
         + np.sin(scales[1] * np.pi * x1_positively_correlated_with_y)
         - scales[2] * x2_negatively_correlated_with_y
         - np.cos(scales[3] * np.pi * x2_negatively_correlated_with_y)
         - scales[4] * x3_negatively_correlated_with_y
         - np.cos(scales[5] * np.pi * x3_negatively_correlated_with_y)
         + zs)
    categorical_features = []
    if x3_to_category:
        categorical_features = [2]
    trainset = lgb.Dataset(x, label=y, categorical_feature=categorical_features, free_raw_data=False)
    return trainset


@pytest.mark.parametrize("test_with_interaction_constraints", [True, False])
def test_monotone_constraints(test_with_interaction_constraints):
    def is_increasing(y):
        return (np.diff(y) >= 0.0).all()

    def is_decreasing(y):
        return (np.diff(y) <= 0.0).all()

    def is_non_monotone(y):
        return (np.diff(y) < 0.0).any() and (np.diff(y) > 0.0).any()

    def is_correctly_constrained(learner, x3_to_category=True):
        iterations = 10
        n = 1000
        variable_x = np.linspace(0, 1, n).reshape((n, 1))
        fixed_xs_values = np.linspace(0, 1, n)
        for i in range(iterations):
            fixed_x = fixed_xs_values[i] * np.ones((n, 1))
            monotonically_increasing_x = np.column_stack((variable_x, fixed_x, fixed_x))
            monotonically_increasing_y = learner.predict(monotonically_increasing_x)
            monotonically_decreasing_x = np.column_stack((fixed_x, variable_x, fixed_x))
            monotonically_decreasing_y = learner.predict(monotonically_decreasing_x)
            non_monotone_x = np.column_stack(
                (
                    fixed_x,
                    fixed_x,
                    categorize(variable_x) if x3_to_category else variable_x,
                )
            )
            non_monotone_y = learner.predict(non_monotone_x)
            if not (
                is_increasing(monotonically_increasing_y)
                and is_decreasing(monotonically_decreasing_y)
                and is_non_monotone(non_monotone_y)
            ):
                return False
        return True

    def are_interactions_enforced(gbm, feature_sets):
        def parse_tree_features(gbm):
            # trees start at position 1.
            tree_str = gbm.model_to_string().split("Tree")[1:]
            feature_sets = []
            for tree in tree_str:
                # split_features are in 4th line.
                features = tree.splitlines()[3].split("=")[1].split(" ")
                features = set(f"Column_{f}" for f in features)
                feature_sets.append(features)
            return np.array(feature_sets)

        def has_interaction(treef):
            n = 0
            for fs in feature_sets:
                if len(treef.intersection(fs)) > 0:
                    n += 1
            return n > 1

        tree_features = parse_tree_features(gbm)
        has_interaction_flag = np.array(
            [has_interaction(treef) for treef in tree_features]
        )

        return not has_interaction_flag.any()

    for test_with_categorical_variable in [True, False]:
        trainset = generate_trainset_for_monotone_constraints_tests(
            test_with_categorical_variable
        )
        for monotone_constraints_method in ["basic", "intermediate", "advanced"]:
            params = {
                "min_data": 20,
                "num_leaves": 20,
                "monotone_constraints": [1, -1, 0],
                "monotone_constraints_method": monotone_constraints_method,
                "use_missing": False,
            }
            if test_with_interaction_constraints:
                params["interaction_constraints"] = [[0], [1], [2]]
            constrained_model = lgb.train(params, trainset)
            assert is_correctly_constrained(
                constrained_model, test_with_categorical_variable
            )
            if test_with_interaction_constraints:
                feature_sets = [["Column_0"], ["Column_1"], "Column_2"]
                assert are_interactions_enforced(constrained_model, feature_sets)


def test_monotone_penalty():
    def are_first_splits_non_monotone(tree, n, monotone_constraints):
        if n <= 0:
            return True
        if "leaf_value" in tree:
            return True
        if monotone_constraints[tree["split_feature"]] != 0:
            return False
        return (are_first_splits_non_monotone(tree["left_child"], n - 1, monotone_constraints)
                and are_first_splits_non_monotone(tree["right_child"], n - 1, monotone_constraints))

    def are_there_monotone_splits(tree, monotone_constraints):
        if "leaf_value" in tree:
            return False
        if monotone_constraints[tree["split_feature"]] != 0:
            return True
        return (are_there_monotone_splits(tree["left_child"], monotone_constraints)
                or are_there_monotone_splits(tree["right_child"], monotone_constraints))

    max_depth = 5
    monotone_constraints = [1, -1, 0]
    penalization_parameter = 2.0
    trainset = generate_trainset_for_monotone_constraints_tests(x3_to_category=False)
    for monotone_constraints_method in ["basic", "intermediate", "advanced"]:
        params = {
            'max_depth': max_depth,
            'monotone_constraints': monotone_constraints,
            'monotone_penalty': penalization_parameter,
            "monotone_constraints_method": monotone_constraints_method,
        }
        constrained_model = lgb.train(params, trainset, 10)
        dumped_model = constrained_model.dump_model()["tree_info"]
        for tree in dumped_model:
            assert are_first_splits_non_monotone(tree["tree_structure"], int(penalization_parameter),
                                                 monotone_constraints)
            assert are_there_monotone_splits(tree["tree_structure"], monotone_constraints)


# test if a penalty as high as the depth indeed prohibits all monotone splits
def test_monotone_penalty_max():
    max_depth = 5
    monotone_constraints = [1, -1, 0]
    penalization_parameter = max_depth
    trainset_constrained_model = generate_trainset_for_monotone_constraints_tests(x3_to_category=False)
    x = trainset_constrained_model.data
    y = trainset_constrained_model.label
    x3_negatively_correlated_with_y = x[:, 2]
    trainset_unconstrained_model = lgb.Dataset(x3_negatively_correlated_with_y.reshape(-1, 1), label=y)
    params_constrained_model = {
        'monotone_constraints': monotone_constraints,
        'monotone_penalty': penalization_parameter,
        "max_depth": max_depth,
        "gpu_use_dp": True,
    }
    params_unconstrained_model = {
        "max_depth": max_depth,
        "gpu_use_dp": True,
    }

    unconstrained_model = lgb.train(params_unconstrained_model, trainset_unconstrained_model, 10)
    unconstrained_model_predictions = unconstrained_model.\
        predict(x3_negatively_correlated_with_y.reshape(-1, 1))

    for monotone_constraints_method in ["basic", "intermediate", "advanced"]:
        params_constrained_model["monotone_constraints_method"] = monotone_constraints_method
        # The penalization is so high that the first 2 features should not be used here
        constrained_model = lgb.train(params_constrained_model, trainset_constrained_model, 10)

        # Check that a very high penalization is the same as not using the features at all
        np.testing.assert_array_equal(constrained_model.predict(x), unconstrained_model_predictions)


def test_max_bin_by_feature():
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
    assert len(np.unique(est.predict(X))) == 100
    params['max_bin_by_feature'] = [2, 100]
    lgb_data = lgb.Dataset(X, label=y)
    est = lgb.train(params, lgb_data, num_boost_round=1)
    assert len(np.unique(est.predict(X))) == 3


def test_small_max_bin():
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
    lgb.train(params, lgb_x, num_boost_round=5)
    x[0, 0] = np.nan
    params['max_bin'] = 3
    lgb_x = lgb.Dataset(x, label=y)
    lgb.train(params, lgb_x, num_boost_round=5)
    np.random.seed()  # reset seed


def test_refit():
    X, y = load_breast_cancer(return_X_y=True)
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
    assert err_pred > new_err_pred


def test_mape_rf():
    X, y = load_boston(return_X_y=True)
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
    assert pred_mean > 20


def test_mape_dart():
    X, y = load_boston(return_X_y=True)
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
    assert pred_mean > 18


def check_constant_features(y_true, expected_pred, more_params):
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
    assert np.allclose(pred, expected_pred)


def test_constant_features_regression():
    params = {
        'objective': 'regression'
    }
    check_constant_features([0.0, 10.0, 0.0, 10.0], 5.0, params)
    check_constant_features([0.0, 1.0, 2.0, 3.0], 1.5, params)
    check_constant_features([-1.0, 1.0, -2.0, 2.0], 0.0, params)


def test_constant_features_binary():
    params = {
        'objective': 'binary'
    }
    check_constant_features([0.0, 10.0, 0.0, 10.0], 0.5, params)
    check_constant_features([0.0, 1.0, 2.0, 3.0], 0.75, params)


def test_constant_features_multiclass():
    params = {
        'objective': 'multiclass',
        'num_class': 3
    }
    check_constant_features([0.0, 1.0, 2.0, 0.0], [0.5, 0.25, 0.25], params)
    check_constant_features([0.0, 1.0, 2.0, 1.0], [0.25, 0.5, 0.25], params)


def test_constant_features_multiclassova():
    params = {
        'objective': 'multiclassova',
        'num_class': 3
    }
    check_constant_features([0.0, 1.0, 2.0, 0.0], [0.5, 0.25, 0.25], params)
    check_constant_features([0.0, 1.0, 2.0, 1.0], [0.25, 0.5, 0.25], params)


def test_fpreproc():
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

    X, y = load_iris(return_X_y=True)
    dataset = lgb.Dataset(X, y, free_raw_data=False)
    params = {'objective': 'multiclass', 'num_class': 3, 'verbose': -1}
    results = lgb.cv(params, dataset, num_boost_round=10, fpreproc=preprocess_data)
    assert 'multi_logloss-mean' in results
    assert len(results['multi_logloss-mean']) == 10


def test_metrics():
    X, y = load_digits(n_class=2, return_X_y=True)
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
        return lgb.cv(params, lgb_train, num_boost_round=2, verbose_eval=False, **kwargs)

    def train_booster(params=params_obj_verbose, **kwargs):
        lgb.train(params, lgb_train,
                  num_boost_round=2,
                  valid_sets=[lgb_valid],
                  evals_result=evals_result,
                  verbose_eval=False, **kwargs)

    # no fobj, no feval
    # default metric
    res = get_cv_result()
    assert len(res) == 2
    assert 'binary_logloss-mean' in res

    # non-default metric in params
    res = get_cv_result(params=params_obj_metric_err_verbose)
    assert len(res) == 2
    assert 'binary_error-mean' in res

    # default metric in args
    res = get_cv_result(metrics='binary_logloss')
    assert len(res) == 2
    assert 'binary_logloss-mean' in res

    # non-default metric in args
    res = get_cv_result(metrics='binary_error')
    assert len(res) == 2
    assert 'binary_error-mean' in res

    # metric in args overwrites one in params
    res = get_cv_result(params=params_obj_metric_inv_verbose, metrics='binary_error')
    assert len(res) == 2
    assert 'binary_error-mean' in res

    # multiple metrics in params
    res = get_cv_result(params=params_obj_metric_multi_verbose)
    assert len(res) == 4
    assert 'binary_logloss-mean' in res
    assert 'binary_error-mean' in res

    # multiple metrics in args
    res = get_cv_result(metrics=['binary_logloss', 'binary_error'])
    assert len(res) == 4
    assert 'binary_logloss-mean' in res
    assert 'binary_error-mean' in res

    # remove default metric by 'None' in list
    res = get_cv_result(metrics=['None'])
    assert len(res) == 0

    # remove default metric by 'None' aliases
    for na_alias in ('None', 'na', 'null', 'custom'):
        res = get_cv_result(metrics=na_alias)
        assert len(res) == 0

    # fobj, no feval
    # no default metric
    res = get_cv_result(params=params_verbose, fobj=dummy_obj)
    assert len(res) == 0

    # metric in params
    res = get_cv_result(params=params_metric_err_verbose, fobj=dummy_obj)
    assert len(res) == 2
    assert 'binary_error-mean' in res

    # metric in args
    res = get_cv_result(params=params_verbose, fobj=dummy_obj, metrics='binary_error')
    assert len(res) == 2
    assert 'binary_error-mean' in res

    # metric in args overwrites its' alias in params
    res = get_cv_result(params=params_metric_inv_verbose, fobj=dummy_obj, metrics='binary_error')
    assert len(res) == 2
    assert 'binary_error-mean' in res

    # multiple metrics in params
    res = get_cv_result(params=params_metric_multi_verbose, fobj=dummy_obj)
    assert len(res) == 4
    assert 'binary_logloss-mean' in res
    assert 'binary_error-mean' in res

    # multiple metrics in args
    res = get_cv_result(params=params_verbose, fobj=dummy_obj,
                        metrics=['binary_logloss', 'binary_error'])
    assert len(res) == 4
    assert 'binary_logloss-mean' in res
    assert 'binary_error-mean' in res

    # no fobj, feval
    # default metric with custom one
    res = get_cv_result(feval=constant_metric)
    assert len(res) == 4
    assert 'binary_logloss-mean' in res
    assert 'error-mean' in res

    # non-default metric in params with custom one
    res = get_cv_result(params=params_obj_metric_err_verbose, feval=constant_metric)
    assert len(res) == 4
    assert 'binary_error-mean' in res
    assert 'error-mean' in res

    # default metric in args with custom one
    res = get_cv_result(metrics='binary_logloss', feval=constant_metric)
    assert len(res) == 4
    assert 'binary_logloss-mean' in res
    assert 'error-mean' in res

    # non-default metric in args with custom one
    res = get_cv_result(metrics='binary_error', feval=constant_metric)
    assert len(res) == 4
    assert 'binary_error-mean' in res
    assert 'error-mean' in res

    # metric in args overwrites one in params, custom one is evaluated too
    res = get_cv_result(params=params_obj_metric_inv_verbose, metrics='binary_error', feval=constant_metric)
    assert len(res) == 4
    assert 'binary_error-mean' in res
    assert 'error-mean' in res

    # multiple metrics in params with custom one
    res = get_cv_result(params=params_obj_metric_multi_verbose, feval=constant_metric)
    assert len(res) == 6
    assert 'binary_logloss-mean' in res
    assert 'binary_error-mean' in res
    assert 'error-mean' in res

    # multiple metrics in args with custom one
    res = get_cv_result(metrics=['binary_logloss', 'binary_error'], feval=constant_metric)
    assert len(res) == 6
    assert 'binary_logloss-mean' in res
    assert 'binary_error-mean' in res
    assert 'error-mean' in res

    # custom metric is evaluated despite 'None' is passed
    res = get_cv_result(metrics=['None'], feval=constant_metric)
    assert len(res) == 2
    assert 'error-mean' in res

    # fobj, feval
    # no default metric, only custom one
    res = get_cv_result(params=params_verbose, fobj=dummy_obj, feval=constant_metric)
    assert len(res) == 2
    assert 'error-mean' in res

    # metric in params with custom one
    res = get_cv_result(params=params_metric_err_verbose, fobj=dummy_obj, feval=constant_metric)
    assert len(res) == 4
    assert 'binary_error-mean' in res
    assert 'error-mean' in res

    # metric in args with custom one
    res = get_cv_result(params=params_verbose, fobj=dummy_obj,
                        feval=constant_metric, metrics='binary_error')
    assert len(res) == 4
    assert 'binary_error-mean' in res
    assert 'error-mean' in res

    # metric in args overwrites one in params, custom one is evaluated too
    res = get_cv_result(params=params_metric_inv_verbose, fobj=dummy_obj,
                        feval=constant_metric, metrics='binary_error')
    assert len(res) == 4
    assert 'binary_error-mean' in res
    assert 'error-mean' in res

    # multiple metrics in params with custom one
    res = get_cv_result(params=params_metric_multi_verbose, fobj=dummy_obj, feval=constant_metric)
    assert len(res) == 6
    assert 'binary_logloss-mean' in res
    assert 'binary_error-mean' in res
    assert 'error-mean' in res

    # multiple metrics in args with custom one
    res = get_cv_result(params=params_verbose, fobj=dummy_obj, feval=constant_metric,
                        metrics=['binary_logloss', 'binary_error'])
    assert len(res) == 6
    assert 'binary_logloss-mean' in res
    assert 'binary_error-mean' in res
    assert 'error-mean' in res

    # custom metric is evaluated despite 'None' is passed
    res = get_cv_result(params=params_metric_none_verbose, fobj=dummy_obj, feval=constant_metric)
    assert len(res) == 2
    assert 'error-mean' in res

    # no fobj, no feval
    # default metric
    train_booster()
    assert len(evals_result['valid_0']) == 1
    assert 'binary_logloss' in evals_result['valid_0']

    # default metric in params
    train_booster(params=params_obj_metric_log_verbose)
    assert len(evals_result['valid_0']) == 1
    assert 'binary_logloss' in evals_result['valid_0']

    # non-default metric in params
    train_booster(params=params_obj_metric_err_verbose)
    assert len(evals_result['valid_0']) == 1
    assert 'binary_error' in evals_result['valid_0']

    # multiple metrics in params
    train_booster(params=params_obj_metric_multi_verbose)
    assert len(evals_result['valid_0']) == 2
    assert 'binary_logloss' in evals_result['valid_0']
    assert 'binary_error' in evals_result['valid_0']

    # remove default metric by 'None' aliases
    for na_alias in ('None', 'na', 'null', 'custom'):
        params = {'objective': 'binary', 'metric': na_alias, 'verbose': -1}
        train_booster(params=params)
        assert len(evals_result) == 0

    # fobj, no feval
    # no default metric
    train_booster(params=params_verbose, fobj=dummy_obj)
    assert len(evals_result) == 0

    # metric in params
    train_booster(params=params_metric_log_verbose, fobj=dummy_obj)
    assert len(evals_result['valid_0']) == 1
    assert 'binary_logloss' in evals_result['valid_0']

    # multiple metrics in params
    train_booster(params=params_metric_multi_verbose, fobj=dummy_obj)
    assert len(evals_result['valid_0']) == 2
    assert 'binary_logloss' in evals_result['valid_0']
    assert 'binary_error' in evals_result['valid_0']

    # no fobj, feval
    # default metric with custom one
    train_booster(feval=constant_metric)
    assert len(evals_result['valid_0']) == 2
    assert 'binary_logloss' in evals_result['valid_0']
    assert 'error' in evals_result['valid_0']

    # default metric in params with custom one
    train_booster(params=params_obj_metric_log_verbose, feval=constant_metric)
    assert len(evals_result['valid_0']) == 2
    assert 'binary_logloss' in evals_result['valid_0']
    assert 'error' in evals_result['valid_0']

    # non-default metric in params with custom one
    train_booster(params=params_obj_metric_err_verbose, feval=constant_metric)
    assert len(evals_result['valid_0']) == 2
    assert 'binary_error' in evals_result['valid_0']
    assert 'error' in evals_result['valid_0']

    # multiple metrics in params with custom one
    train_booster(params=params_obj_metric_multi_verbose, feval=constant_metric)
    assert len(evals_result['valid_0']) == 3
    assert 'binary_logloss' in evals_result['valid_0']
    assert 'binary_error' in evals_result['valid_0']
    assert 'error' in evals_result['valid_0']

    # custom metric is evaluated despite 'None' is passed
    train_booster(params=params_obj_metric_none_verbose, feval=constant_metric)
    assert len(evals_result) == 1
    assert 'error' in evals_result['valid_0']

    # fobj, feval
    # no default metric, only custom one
    train_booster(params=params_verbose, fobj=dummy_obj, feval=constant_metric)
    assert len(evals_result['valid_0']) == 1
    assert 'error' in evals_result['valid_0']

    # metric in params with custom one
    train_booster(params=params_metric_log_verbose, fobj=dummy_obj, feval=constant_metric)
    assert len(evals_result['valid_0']) == 2
    assert 'binary_logloss' in evals_result['valid_0']
    assert 'error' in evals_result['valid_0']

    # multiple metrics in params with custom one
    train_booster(params=params_metric_multi_verbose, fobj=dummy_obj, feval=constant_metric)
    assert len(evals_result['valid_0']) == 3
    assert 'binary_logloss' in evals_result['valid_0']
    assert 'binary_error' in evals_result['valid_0']
    assert 'error' in evals_result['valid_0']

    # custom metric is evaluated despite 'None' is passed
    train_booster(params=params_metric_none_verbose, fobj=dummy_obj, feval=constant_metric)
    assert len(evals_result) == 1
    assert 'error' in evals_result['valid_0']

    X, y = load_digits(n_class=3, return_X_y=True)
    lgb_train = lgb.Dataset(X, y, silent=True)

    obj_multi_aliases = ['multiclass', 'softmax', 'multiclassova', 'multiclass_ova', 'ova', 'ovr']
    for obj_multi_alias in obj_multi_aliases:
        params_obj_class_3_verbose = {'objective': obj_multi_alias, 'num_class': 3, 'verbose': -1}
        params_obj_class_1_verbose = {'objective': obj_multi_alias, 'num_class': 1, 'verbose': -1}
        params_obj_verbose = {'objective': obj_multi_alias, 'verbose': -1}
        # multiclass default metric
        res = get_cv_result(params_obj_class_3_verbose)
        assert len(res) == 2
        assert 'multi_logloss-mean' in res
        # multiclass default metric with custom one
        res = get_cv_result(params_obj_class_3_verbose, feval=constant_metric)
        assert len(res) == 4
        assert 'multi_logloss-mean' in res
        assert 'error-mean' in res
        # multiclass metric alias with custom one for custom objective
        res = get_cv_result(params_obj_class_3_verbose, fobj=dummy_obj, feval=constant_metric)
        assert len(res) == 2
        assert 'error-mean' in res
        # no metric for invalid class_num
        res = get_cv_result(params_obj_class_1_verbose, fobj=dummy_obj)
        assert len(res) == 0
        # custom metric for invalid class_num
        res = get_cv_result(params_obj_class_1_verbose, fobj=dummy_obj, feval=constant_metric)
        assert len(res) == 2
        assert 'error-mean' in res
        # multiclass metric alias with custom one with invalid class_num
        with pytest.raises(lgb.basic.LightGBMError):
            get_cv_result(params_obj_class_1_verbose, metrics=obj_multi_alias,
                          fobj=dummy_obj, feval=constant_metric)
        # multiclass default metric without num_class
        with pytest.raises(lgb.basic.LightGBMError):
            get_cv_result(params_obj_verbose)
        for metric_multi_alias in obj_multi_aliases + ['multi_logloss']:
            # multiclass metric alias
            res = get_cv_result(params_obj_class_3_verbose, metrics=metric_multi_alias)
            assert len(res) == 2
            assert 'multi_logloss-mean' in res
        # multiclass metric
        res = get_cv_result(params_obj_class_3_verbose, metrics='multi_error')
        assert len(res) == 2
        assert 'multi_error-mean' in res
        # non-valid metric for multiclass objective
        with pytest.raises(lgb.basic.LightGBMError):
            get_cv_result(params_obj_class_3_verbose, metrics='binary_logloss')
    params_class_3_verbose = {'num_class': 3, 'verbose': -1}
    # non-default num_class for default objective
    with pytest.raises(lgb.basic.LightGBMError):
        get_cv_result(params_class_3_verbose)
    # no metric with non-default num_class for custom objective
    res = get_cv_result(params_class_3_verbose, fobj=dummy_obj)
    assert len(res) == 0
    for metric_multi_alias in obj_multi_aliases + ['multi_logloss']:
        # multiclass metric alias for custom objective
        res = get_cv_result(params_class_3_verbose, metrics=metric_multi_alias, fobj=dummy_obj)
        assert len(res) == 2
        assert 'multi_logloss-mean' in res
    # multiclass metric for custom objective
    res = get_cv_result(params_class_3_verbose, metrics='multi_error', fobj=dummy_obj)
    assert len(res) == 2
    assert 'multi_error-mean' in res
    # binary metric with non-default num_class for custom objective
    with pytest.raises(lgb.basic.LightGBMError):
        get_cv_result(params_class_3_verbose, metrics='binary_error', fobj=dummy_obj)


def test_multiple_feval_train():
    X, y = load_breast_cancer(return_X_y=True)

    params = {'verbose': -1, 'objective': 'binary', 'metric': 'binary_logloss'}

    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2)

    train_dataset = lgb.Dataset(data=X_train, label=y_train, silent=True)
    validation_dataset = lgb.Dataset(data=X_validation, label=y_validation, reference=train_dataset, silent=True)
    evals_result = {}
    lgb.train(
        params=params,
        train_set=train_dataset,
        valid_sets=validation_dataset,
        num_boost_round=5,
        feval=[constant_metric, decreasing_metric],
        evals_result=evals_result)

    assert len(evals_result['valid_0']) == 3
    assert 'binary_logloss' in evals_result['valid_0']
    assert 'error' in evals_result['valid_0']
    assert 'decreasing_metric' in evals_result['valid_0']


def test_multiple_feval_cv():
    X, y = load_breast_cancer(return_X_y=True)

    params = {'verbose': -1, 'objective': 'binary', 'metric': 'binary_logloss'}

    train_dataset = lgb.Dataset(data=X, label=y, silent=True)

    cv_results = lgb.cv(
        params=params,
        train_set=train_dataset,
        num_boost_round=5,
        feval=[constant_metric, decreasing_metric])

    # Expect three metrics but mean and stdv for each metric
    assert len(cv_results) == 6
    assert 'binary_logloss-mean' in cv_results
    assert 'error-mean' in cv_results
    assert 'decreasing_metric-mean' in cv_results
    assert 'binary_logloss-stdv' in cv_results
    assert 'error-stdv' in cv_results
    assert 'decreasing_metric-stdv' in cv_results


@pytest.mark.skipif(psutil.virtual_memory().available / 1024 / 1024 / 1024 < 3, reason='not enough RAM')
def test_model_size():
    X, y = load_boston(return_X_y=True)
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
        assert len(new_model_str) > 2**31
        bst.model_from_string(new_model_str, verbose=False)
        assert bst.num_trees() == total_trees
        y_pred_new = bst.predict(X, num_iteration=2)
        np.testing.assert_allclose(y_pred, y_pred_new)
    except MemoryError:
        pytest.skipTest('not enough RAM')


def test_get_split_value_histogram():
    X, y = load_boston(return_X_y=True)
    lgb_train = lgb.Dataset(X, y, categorical_feature=[2])
    gbm = lgb.train({'verbose': -1}, lgb_train, num_boost_round=20)
    # test XGBoost-style return value
    params = {'feature': 0, 'xgboost_style': True}
    assert gbm.get_split_value_histogram(**params).shape == (9, 2)
    assert gbm.get_split_value_histogram(bins=999, **params).shape == (9, 2)
    assert gbm.get_split_value_histogram(bins=-1, **params).shape == (1, 2)
    assert gbm.get_split_value_histogram(bins=0, **params).shape == (1, 2)
    assert gbm.get_split_value_histogram(bins=1, **params).shape == (1, 2)
    assert gbm.get_split_value_histogram(bins=2, **params).shape == (2, 2)
    assert gbm.get_split_value_histogram(bins=6, **params).shape == (5, 2)
    assert gbm.get_split_value_histogram(bins=7, **params).shape == (6, 2)
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
    assert len(hist) == 23
    assert len(bins) == 24
    hist, bins = gbm.get_split_value_histogram(0, bins=999)
    assert len(hist) == 999
    assert len(bins) == 1000
    with pytest.raises(ValueError):
        gbm.get_split_value_histogram(0, bins=-1)
    with pytest.raises(ValueError):
        gbm.get_split_value_histogram(0, bins=0)
    hist, bins = gbm.get_split_value_histogram(0, bins=1)
    assert len(hist) == 1
    assert len(bins) == 2
    hist, bins = gbm.get_split_value_histogram(0, bins=2)
    assert len(hist) == 2
    assert len(bins) == 3
    hist, bins = gbm.get_split_value_histogram(0, bins=6)
    assert len(hist) == 6
    assert len(bins) == 7
    hist, bins = gbm.get_split_value_histogram(0, bins=7)
    assert len(hist) == 7
    assert len(bins) == 8
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
    with pytest.raises(lgb.basic.LightGBMError):
        gbm.get_split_value_histogram(2)


def test_early_stopping_for_only_first_metric():

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
        assert assumed_iteration == gbm.best_iteration

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
        assert assumed_iteration == len(ret[list(ret.keys())[0]])

    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test1, X_test2, y_test1, y_test2 = train_test_split(X_test, y_test, test_size=0.5, random_state=73)
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_valid1 = lgb.Dataset(X_test1, y_test1, reference=lgb_train)
    lgb_valid2 = lgb.Dataset(X_test2, y_test2, reference=lgb_train)

    iter_valid1_l1 = 3
    iter_valid1_l2 = 14
    iter_valid2_l1 = 2
    iter_valid2_l2 = 15
    assert len(set([iter_valid1_l1, iter_valid1_l2, iter_valid2_l1, iter_valid2_l2])) == 4
    iter_min_l1 = min([iter_valid1_l1, iter_valid2_l1])
    iter_min_l2 = min([iter_valid1_l2, iter_valid2_l2])
    iter_min_valid1 = min([iter_valid1_l1, iter_valid1_l2])

    iter_cv_l1 = 4
    iter_cv_l2 = 12
    assert len(set([iter_cv_l1, iter_cv_l2])) == 2
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


def test_node_level_subcol():
    X, y = load_breast_cancer(return_X_y=True)
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
    assert ret < 0.14
    assert evals_result['valid_0']['binary_logloss'][-1] == pytest.approx(ret)
    params['feature_fraction'] = 0.5
    gbm2 = lgb.train(params, lgb_train, num_boost_round=25)
    ret2 = log_loss(y_test, gbm2.predict(X_test))
    assert ret != ret2


def test_forced_bins():
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
              'verbose': -1}
    lgb_x = lgb.Dataset(x, label=y)
    est = lgb.train(params, lgb_x, num_boost_round=20)
    new_x = np.zeros((3, x.shape[1]))
    new_x[:, 0] = [0.31, 0.37, 0.41]
    new_x[:, 1] = [0, 0, 0]
    predicted = est.predict(new_x)
    assert len(np.unique(predicted)) == 3
    new_x[:, 0] = [0, 0, 0]
    new_x[:, 1] = [-0.9, -0.6, -0.3]
    predicted = est.predict(new_x)
    assert len(np.unique(predicted)) == 1
    params['forcedbins_filename'] = ''
    lgb_x = lgb.Dataset(x, label=y)
    est = lgb.train(params, lgb_x, num_boost_round=20)
    predicted = est.predict(new_x)
    assert len(np.unique(predicted)) == 3
    params['forcedbins_filename'] = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                 '../../examples/regression/forced_bins2.json')
    params['max_bin'] = 11
    lgb_x = lgb.Dataset(x[:, :1], label=y)
    est = lgb.train(params, lgb_x, num_boost_round=50)
    predicted = est.predict(x[1:, :1])
    _, counts = np.unique(predicted, return_counts=True)
    assert min(counts) >= 9
    assert max(counts) <= 11


def test_binning_same_sign():
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
    est = lgb.train(params, lgb_x, num_boost_round=20)
    new_x = np.zeros((3, 2))
    new_x[:, 0] = [-1, 0, 1]
    predicted = est.predict(new_x)
    assert predicted[0] == pytest.approx(predicted[1])
    assert predicted[1] != pytest.approx(predicted[2])
    new_x = np.zeros((3, 2))
    new_x[:, 1] = [-1, 0, 1]
    predicted = est.predict(new_x)
    assert predicted[0] != pytest.approx(predicted[1])
    assert predicted[1] == pytest.approx(predicted[2])


def test_dataset_update_params():
    default_params = {"max_bin": 100,
                      "max_bin_by_feature": [20, 10],
                      "bin_construct_sample_cnt": 10000,
                      "min_data_in_bin": 1,
                      "use_missing": False,
                      "zero_as_missing": False,
                      "categorical_feature": [0],
                      "feature_pre_filter": True,
                      "pre_partition": False,
                      "enable_bundle": True,
                      "data_random_seed": 0,
                      "is_enable_sparse": True,
                      "header": True,
                      "two_round": True,
                      "label_column": 0,
                      "weight_column": 0,
                      "group_column": 0,
                      "ignore_column": 0,
                      "min_data_in_leaf": 10,
                      "linear_tree": False,
                      "verbose": -1}
    unchangeable_params = {"max_bin": 150,
                           "max_bin_by_feature": [30, 5],
                           "bin_construct_sample_cnt": 5000,
                           "min_data_in_bin": 2,
                           "use_missing": True,
                           "zero_as_missing": True,
                           "categorical_feature": [0, 1],
                           "feature_pre_filter": False,
                           "pre_partition": True,
                           "enable_bundle": False,
                           "data_random_seed": 1,
                           "is_enable_sparse": False,
                           "header": False,
                           "two_round": False,
                           "label_column": 1,
                           "weight_column": 1,
                           "group_column": 1,
                           "ignore_column": 1,
                           "forcedbins_filename": "/some/path/forcedbins.json",
                           "min_data_in_leaf": 2,
                           "linear_tree": True}
    X = np.random.random((100, 2))
    y = np.random.random(100)

    # decreasing without freeing raw data is allowed
    lgb_data = lgb.Dataset(X, y, params=default_params, free_raw_data=False).construct()
    default_params["min_data_in_leaf"] -= 1
    lgb.train(default_params, lgb_data, num_boost_round=3)

    # decreasing before lazy init is allowed
    lgb_data = lgb.Dataset(X, y, params=default_params)
    default_params["min_data_in_leaf"] -= 1
    lgb.train(default_params, lgb_data, num_boost_round=3)

    # increasing is allowed
    default_params["min_data_in_leaf"] += 2
    lgb.train(default_params, lgb_data, num_boost_round=3)

    # decreasing with disabled filter is allowed
    default_params["feature_pre_filter"] = False
    lgb_data = lgb.Dataset(X, y, params=default_params).construct()
    default_params["min_data_in_leaf"] -= 4
    lgb.train(default_params, lgb_data, num_boost_round=3)

    # decreasing with enabled filter is disallowed;
    # also changes of other params are disallowed
    default_params["feature_pre_filter"] = True
    lgb_data = lgb.Dataset(X, y, params=default_params).construct()
    for key, value in unchangeable_params.items():
        new_params = default_params.copy()
        new_params[key] = value
        err_msg = ("Reducing `min_data_in_leaf` with `feature_pre_filter=true` may cause *"
                   if key == "min_data_in_leaf"
                   else "Cannot change {} *".format(key if key != "forcedbins_filename"
                                                    else "forced bins"))
        with np.testing.assert_raises_regex(lgb.basic.LightGBMError, err_msg):
            lgb.train(new_params, lgb_data, num_boost_round=3)


def test_dataset_params_with_reference():
    default_params = {"max_bin": 100}
    X = np.random.random((100, 2))
    y = np.random.random(100)
    X_val = np.random.random((100, 2))
    y_val = np.random.random(100)
    lgb_train = lgb.Dataset(X, y, params=default_params, free_raw_data=False).construct()
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train, free_raw_data=False).construct()
    assert lgb_train.get_params() == default_params
    assert lgb_val.get_params() == default_params
    lgb.train(default_params, lgb_train, valid_sets=[lgb_val])


def test_extra_trees():
    # check extra trees increases regularization
    X, y = load_boston(return_X_y=True)
    lgb_x = lgb.Dataset(X, label=y)
    params = {'objective': 'regression',
              'num_leaves': 32,
              'verbose': -1,
              'extra_trees': False,
              'seed': 0}
    est = lgb.train(params, lgb_x, num_boost_round=10)
    predicted = est.predict(X)
    err = mean_squared_error(y, predicted)
    params['extra_trees'] = True
    est = lgb.train(params, lgb_x, num_boost_round=10)
    predicted_new = est.predict(X)
    err_new = mean_squared_error(y, predicted_new)
    assert err < err_new


def test_path_smoothing():
    # check path smoothing increases regularization
    X, y = load_boston(return_X_y=True)
    lgb_x = lgb.Dataset(X, label=y)
    params = {'objective': 'regression',
              'num_leaves': 32,
              'verbose': -1,
              'seed': 0}
    est = lgb.train(params, lgb_x, num_boost_round=10)
    predicted = est.predict(X)
    err = mean_squared_error(y, predicted)
    params['path_smooth'] = 1
    est = lgb.train(params, lgb_x, num_boost_round=10)
    predicted_new = est.predict(X)
    err_new = mean_squared_error(y, predicted_new)
    assert err < err_new


def test_trees_to_dataframe():
    pytest.importorskip("pandas")

    def _imptcs_to_numpy(X, impcts_dict):
        cols = ['Column_' + str(i) for i in range(X.shape[1])]
        return [impcts_dict.get(col, 0.) for col in cols]

    X, y = load_breast_cancer(return_X_y=True)
    data = lgb.Dataset(X, label=y)
    num_trees = 10
    bst = lgb.train({"objective": "binary", "verbose": -1}, data, num_trees)
    tree_df = bst.trees_to_dataframe()
    split_dict = (tree_df[~tree_df['split_gain'].isnull()]
                  .groupby('split_feature')
                  .size()
                  .to_dict())

    gains_dict = (tree_df
                  .groupby('split_feature')['split_gain']
                  .sum()
                  .to_dict())

    tree_split = _imptcs_to_numpy(X, split_dict)
    tree_gains = _imptcs_to_numpy(X, gains_dict)
    mod_split = bst.feature_importance('split')
    mod_gains = bst.feature_importance('gain')
    num_trees_from_df = tree_df['tree_index'].nunique()
    obs_counts_from_df = tree_df.loc[tree_df['node_depth'] == 1, 'count'].values

    np.testing.assert_equal(tree_split, mod_split)
    np.testing.assert_allclose(tree_gains, mod_gains)
    assert num_trees_from_df == num_trees
    np.testing.assert_equal(obs_counts_from_df, len(y))

    # test edge case with one leaf
    X = np.ones((10, 2))
    y = np.random.rand(10)
    data = lgb.Dataset(X, label=y)
    bst = lgb.train({"objective": "binary", "verbose": -1}, data, num_trees)
    tree_df = bst.trees_to_dataframe()

    assert len(tree_df) == 1
    assert tree_df.loc[0, 'tree_index'] == 0
    assert tree_df.loc[0, 'node_depth'] == 1
    assert tree_df.loc[0, 'node_index'] == "0-L0"
    assert tree_df.loc[0, 'value'] is not None
    for col in ('left_child', 'right_child', 'parent_index', 'split_feature',
                'split_gain', 'threshold', 'decision_type', 'missing_direction',
                'missing_type', 'weight', 'count'):
        assert tree_df.loc[0, col] is None


def test_interaction_constraints():
    X, y = load_boston(return_X_y=True)
    num_features = X.shape[1]
    train_data = lgb.Dataset(X, label=y)
    # check that constraint containing all features is equivalent to no constraint
    params = {'verbose': -1,
              'seed': 0}
    est = lgb.train(params, train_data, num_boost_round=10)
    pred1 = est.predict(X)
    est = lgb.train(dict(params, interaction_constraints=[list(range(num_features))]), train_data,
                    num_boost_round=10)
    pred2 = est.predict(X)
    np.testing.assert_allclose(pred1, pred2)
    # check that constraint partitioning the features reduces train accuracy
    est = lgb.train(dict(params, interaction_constraints=[list(range(num_features // 2)),
                                                          list(range(num_features // 2, num_features))]),
                    train_data, num_boost_round=10)
    pred3 = est.predict(X)
    assert mean_squared_error(y, pred1) < mean_squared_error(y, pred3)
    # check that constraints consisting of single features reduce accuracy further
    est = lgb.train(dict(params, interaction_constraints=[[i] for i in range(num_features)]), train_data,
                    num_boost_round=10)
    pred4 = est.predict(X)
    assert mean_squared_error(y, pred3) < mean_squared_error(y, pred4)
    # test that interaction constraints work when not all features are used
    X = np.concatenate([np.zeros((X.shape[0], 1)), X], axis=1)
    num_features = X.shape[1]
    train_data = lgb.Dataset(X, label=y)
    est = lgb.train(dict(params, interaction_constraints=[[0] + list(range(2, num_features)),
                                                          [1] + list(range(2, num_features))]),
                    train_data, num_boost_round=10)


def test_linear_trees(tmp_path):
    # check that setting linear_tree=True fits better than ordinary trees when data has linear relationship
    np.random.seed(0)
    x = np.arange(0, 100, 0.1)
    y = 2 * x + np.random.normal(0, 0.1, len(x))
    x = x[:, np.newaxis]
    lgb_train = lgb.Dataset(x, label=y)
    params = {'verbose': -1,
              'metric': 'mse',
              'seed': 0,
              'num_leaves': 2}
    est = lgb.train(params, lgb_train, num_boost_round=10)
    pred1 = est.predict(x)
    lgb_train = lgb.Dataset(x, label=y)
    res = {}
    est = lgb.train(dict(params, linear_tree=True), lgb_train, num_boost_round=10, evals_result=res,
                    valid_sets=[lgb_train], valid_names=['train'])
    pred2 = est.predict(x)
    assert res['train']['l2'][-1] == pytest.approx(mean_squared_error(y, pred2), abs=1e-1)
    assert mean_squared_error(y, pred2) < mean_squared_error(y, pred1)
    # test again with nans in data
    x[:10] = np.nan
    lgb_train = lgb.Dataset(x, label=y)
    est = lgb.train(params, lgb_train, num_boost_round=10)
    pred1 = est.predict(x)
    lgb_train = lgb.Dataset(x, label=y)
    res = {}
    est = lgb.train(dict(params, linear_tree=True), lgb_train, num_boost_round=10, evals_result=res,
                    valid_sets=[lgb_train], valid_names=['train'])
    pred2 = est.predict(x)
    assert res['train']['l2'][-1] == pytest.approx(mean_squared_error(y, pred2), abs=1e-1)
    assert mean_squared_error(y, pred2) < mean_squared_error(y, pred1)
    # test again with bagging
    res = {}
    est = lgb.train(dict(params, linear_tree=True, subsample=0.8, bagging_freq=1), lgb_train,
                    num_boost_round=10, evals_result=res, valid_sets=[lgb_train], valid_names=['train'])
    pred = est.predict(x)
    assert res['train']['l2'][-1] == pytest.approx(mean_squared_error(y, pred), abs=1e-1)
    # test with a feature that has only one non-nan value
    x = np.concatenate([np.ones([x.shape[0], 1]), x], 1)
    x[500:, 1] = np.nan
    y[500:] += 10
    lgb_train = lgb.Dataset(x, label=y)
    res = {}
    est = lgb.train(dict(params, linear_tree=True, subsample=0.8, bagging_freq=1), lgb_train,
                    num_boost_round=10, evals_result=res, valid_sets=[lgb_train], valid_names=['train'])
    pred = est.predict(x)
    assert res['train']['l2'][-1] == pytest.approx(mean_squared_error(y, pred), abs=1e-1)
    # test with a categorical feature
    x[:250, 0] = 0
    y[:250] += 10
    lgb_train = lgb.Dataset(x, label=y)
    est = lgb.train(dict(params, linear_tree=True, subsample=0.8, bagging_freq=1), lgb_train,
                    num_boost_round=10, categorical_feature=[0])
    # test refit: same results on same data
    est2 = est.refit(x, label=y)
    p1 = est.predict(x)
    p2 = est2.predict(x)
    assert np.mean(np.abs(p1 - p2)) < 2

    # test refit with save and load
    temp_model = str(tmp_path / "temp_model.txt")
    est.save_model(temp_model)
    est2 = lgb.Booster(model_file=temp_model)
    est2 = est2.refit(x, label=y)
    p1 = est.predict(x)
    p2 = est2.predict(x)
    assert np.mean(np.abs(p1 - p2)) < 2
    # test refit: different results training on different data
    est3 = est.refit(x[:100, :], label=y[:100])
    p3 = est3.predict(x)
    assert np.mean(np.abs(p2 - p1)) > np.abs(np.max(p3 - p1))
    # test when num_leaves - 1 < num_features and when num_leaves - 1 > num_features
    X_train, _, y_train, _ = train_test_split(*load_breast_cancer(return_X_y=True), test_size=0.1, random_state=2)
    params = {'linear_tree': True,
              'verbose': -1,
              'metric': 'mse',
              'seed': 0}
    train_data = lgb.Dataset(X_train, label=y_train, params=dict(params, num_leaves=2))
    est = lgb.train(params, train_data, num_boost_round=10, categorical_feature=[0])
    train_data = lgb.Dataset(X_train, label=y_train, params=dict(params, num_leaves=60))
    est = lgb.train(params, train_data, num_boost_round=10, categorical_feature=[0])


def test_save_and_load_linear(tmp_path):
    X_train, X_test, y_train, y_test = train_test_split(*load_breast_cancer(return_X_y=True), test_size=0.1,
                                                        random_state=2)
    X_train = np.concatenate([np.ones((X_train.shape[0], 1)), X_train], 1)
    X_train[:X_train.shape[0] // 2, 0] = 0
    y_train[:X_train.shape[0] // 2] = 1
    params = {'linear_tree': True}
    train_data_1 = lgb.Dataset(X_train, label=y_train, params=params)
    est_1 = lgb.train(params, train_data_1, num_boost_round=10, categorical_feature=[0])
    pred_1 = est_1.predict(X_train)

    tmp_dataset = str(tmp_path / 'temp_dataset.bin')
    train_data_1.save_binary(tmp_dataset)
    train_data_2 = lgb.Dataset(tmp_dataset)
    est_2 = lgb.train(params, train_data_2, num_boost_round=10)
    pred_2 = est_2.predict(X_train)
    np.testing.assert_allclose(pred_1, pred_2)

    model_file = str(tmp_path / 'model.txt')
    est_2.save_model(model_file)
    est_3 = lgb.Booster(model_file=model_file)
    pred_3 = est_3.predict(X_train)
    np.testing.assert_allclose(pred_2, pred_3)


def test_linear_single_leaf():
    X_train, y_train = load_breast_cancer(return_X_y=True)
    train_data = lgb.Dataset(X_train, label=y_train)
    params = {
        "objective": "binary",
        "linear_tree": True,
        "min_sum_hessian": 5000
    }
    bst = lgb.train(params, train_data, num_boost_round=5)
    y_pred = bst.predict(X_train)
    assert log_loss(y_train, y_pred) < 0.661


def test_predict_with_start_iteration():
    def inner_test(X, y, params, early_stopping_rounds):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test)
        booster = lgb.train(params, train_data, num_boost_round=50, early_stopping_rounds=early_stopping_rounds,
                            valid_sets=[valid_data])

        # test that the predict once with all iterations equals summed results with start_iteration and num_iteration
        all_pred = booster.predict(X, raw_score=True)
        all_pred_contrib = booster.predict(X, pred_contrib=True)
        steps = [10, 12]
        for step in steps:
            pred = np.zeros_like(all_pred)
            pred_contrib = np.zeros_like(all_pred_contrib)
            for start_iter in range(0, 50, step):
                pred += booster.predict(X, start_iteration=start_iter, num_iteration=step, raw_score=True)
                pred_contrib += booster.predict(X, start_iteration=start_iter, num_iteration=step, pred_contrib=True)
            np.testing.assert_allclose(all_pred, pred)
            np.testing.assert_allclose(all_pred_contrib, pred_contrib)
        # test the case where start_iteration <= 0, and num_iteration is None
        pred1 = booster.predict(X, start_iteration=-1)
        pred2 = booster.predict(X, num_iteration=booster.best_iteration)
        np.testing.assert_allclose(pred1, pred2)

        # test the case where start_iteration > 0, and num_iteration <= 0
        pred4 = booster.predict(X, start_iteration=10, num_iteration=-1)
        pred5 = booster.predict(X, start_iteration=10, num_iteration=90)
        pred6 = booster.predict(X, start_iteration=10, num_iteration=0)
        np.testing.assert_allclose(pred4, pred5)
        np.testing.assert_allclose(pred4, pred6)

        # test the case where start_iteration > 0, and num_iteration <= 0, with pred_leaf=True
        pred4 = booster.predict(X, start_iteration=10, num_iteration=-1, pred_leaf=True)
        pred5 = booster.predict(X, start_iteration=10, num_iteration=40, pred_leaf=True)
        pred6 = booster.predict(X, start_iteration=10, num_iteration=0, pred_leaf=True)
        np.testing.assert_allclose(pred4, pred5)
        np.testing.assert_allclose(pred4, pred6)

        # test the case where start_iteration > 0, and num_iteration <= 0, with pred_contrib=True
        pred4 = booster.predict(X, start_iteration=10, num_iteration=-1, pred_contrib=True)
        pred5 = booster.predict(X, start_iteration=10, num_iteration=40, pred_contrib=True)
        pred6 = booster.predict(X, start_iteration=10, num_iteration=0, pred_contrib=True)
        np.testing.assert_allclose(pred4, pred5)
        np.testing.assert_allclose(pred4, pred6)

    # test for regression
    X, y = load_boston(return_X_y=True)
    params = {
        'objective': 'regression',
        'verbose': -1,
        'metric': 'l2',
        'learning_rate': 0.5
    }
    # test both with and without early stopping
    inner_test(X, y, params, early_stopping_rounds=1)
    inner_test(X, y, params, early_stopping_rounds=5)
    inner_test(X, y, params, early_stopping_rounds=None)

    # test for multi-class
    X, y = load_iris(return_X_y=True)
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'verbose': -1,
        'metric': 'multi_error'
    }
    # test both with and without early stopping
    inner_test(X, y, params, early_stopping_rounds=1)
    inner_test(X, y, params, early_stopping_rounds=5)
    inner_test(X, y, params, early_stopping_rounds=None)

    # test for binary
    X, y = load_breast_cancer(return_X_y=True)
    params = {
        'objective': 'binary',
        'verbose': -1,
        'metric': 'auc'
    }
    # test both with and without early stopping
    inner_test(X, y, params, early_stopping_rounds=1)
    inner_test(X, y, params, early_stopping_rounds=5)
    inner_test(X, y, params, early_stopping_rounds=None)


def test_average_precision_metric():
    # test against sklearn average precision metric
    X, y = load_breast_cancer(return_X_y=True)
    params = {
        'objective': 'binary',
        'metric': 'average_precision',
        'verbose': -1
    }
    res = {}
    lgb_X = lgb.Dataset(X, label=y)
    est = lgb.train(params, lgb_X, num_boost_round=10, valid_sets=[lgb_X], evals_result=res)
    ap = res['training']['average_precision'][-1]
    pred = est.predict(X)
    sklearn_ap = average_precision_score(y, pred)
    assert ap == pytest.approx(sklearn_ap)
    # test that average precision is 1 where model predicts perfectly
    y = y.copy()
    y[:] = 1
    lgb_X = lgb.Dataset(X, label=y)
    lgb.train(params, lgb_X, num_boost_round=1, valid_sets=[lgb_X], evals_result=res)
    assert res['training']['average_precision'][-1] == pytest.approx(1)


def test_reset_params_works_with_metric_num_class_and_boosting():
    X, y = load_breast_cancer(return_X_y=True)
    dataset_params = {"max_bin": 150}
    booster_params = {
        'objective': 'multiclass',
        'max_depth': 4,
        'bagging_fraction': 0.8,
        'metric': ['multi_logloss', 'multi_error'],
        'boosting': 'gbdt',
        'num_class': 5
    }
    dtrain = lgb.Dataset(X, y, params=dataset_params)
    bst = lgb.Booster(
        params=booster_params,
        train_set=dtrain
    )

    expected_params = dict(dataset_params, **booster_params)
    assert bst.params == expected_params

    booster_params['bagging_fraction'] += 0.1
    new_bst = bst.reset_parameter(booster_params)

    expected_params = dict(dataset_params, **booster_params)
    assert bst.params == expected_params
    assert new_bst.params == expected_params


def test_dump_model():
    X, y = load_breast_cancer(return_X_y=True)
    train_data = lgb.Dataset(X, label=y)
    params = {
        "objective": "binary",
        "verbose": -1
    }
    bst = lgb.train(params, train_data, num_boost_round=5)
    dumped_model_str = str(bst.dump_model(5, 0))
    assert "leaf_features" not in dumped_model_str
    assert "leaf_coeff" not in dumped_model_str
    assert "leaf_const" not in dumped_model_str
    assert "leaf_value" in dumped_model_str
    assert "leaf_count" in dumped_model_str
    params['linear_tree'] = True
    train_data = lgb.Dataset(X, label=y)
    bst = lgb.train(params, train_data, num_boost_round=5)
    dumped_model_str = str(bst.dump_model(5, 0))
    assert "leaf_features" in dumped_model_str
    assert "leaf_coeff" in dumped_model_str
    assert "leaf_const" in dumped_model_str
    assert "leaf_value" in dumped_model_str
    assert "leaf_count" in dumped_model_str
