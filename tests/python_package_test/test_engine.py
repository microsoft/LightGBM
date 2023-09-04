# coding: utf-8
import copy
import itertools
import json
import math
import pickle
import platform
import random
import re
from os import getenv
from pathlib import Path
from shutil import copyfile

import numpy as np
import psutil
import pytest
from scipy.sparse import csr_matrix, isspmatrix_csc, isspmatrix_csr
from sklearn.datasets import load_svmlight_file, make_blobs, make_multilabel_classification
from sklearn.metrics import average_precision_score, log_loss, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import GroupKFold, TimeSeriesSplit, train_test_split

import lightgbm as lgb
from lightgbm.compat import PANDAS_INSTALLED, pd_DataFrame, pd_Series

from .utils import (SERIALIZERS, dummy_obj, load_breast_cancer, load_digits, load_iris, logistic_sigmoid,
                    make_synthetic_regression, mse_obj, pickle_and_unpickle_object, sklearn_multiclass_custom_objective,
                    softmax)

decreasing_generator = itertools.count(0, -1)


def logloss_obj(preds, train_data):
    y_true = train_data.get_label()
    y_pred = logistic_sigmoid(preds)
    grad = y_pred - y_true
    hess = y_pred * (1.0 - y_pred)
    return grad, hess


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
    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=20,
        valid_sets=lgb_eval,
        callbacks=[lgb.record_evaluation(evals_result)]
    )
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
    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=50,
        valid_sets=lgb_eval,
        callbacks=[lgb.record_evaluation(evals_result)]
    )
    ret = log_loss(y_test, gbm.predict(X_test))
    assert ret < 0.19
    assert evals_result['valid_0']['binary_logloss'][-1] == pytest.approx(ret)


@pytest.mark.parametrize('objective', ['regression', 'regression_l1', 'huber', 'fair', 'poisson', 'quantile'])
def test_regression(objective):
    X, y = make_synthetic_regression()
    y = np.abs(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    params = {
        'objective': objective,
        'metric': 'l2',
        'verbose': -1
    }
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    evals_result = {}
    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=50,
        valid_sets=lgb_eval,
        callbacks=[lgb.record_evaluation(evals_result)]
    )
    ret = mean_squared_error(y_test, gbm.predict(X_test))
    if objective == 'huber':
        assert ret < 430
    elif objective == 'fair':
        assert ret < 296
    elif objective == 'poisson':
        assert ret < 193
    elif objective == 'quantile':
        assert ret < 1311
    else:
        assert ret < 343
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
    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=20,
        valid_sets=lgb_eval,
        callbacks=[lgb.record_evaluation(evals_result)]
    )
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
    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=20,
        valid_sets=lgb_eval,
        callbacks=[lgb.record_evaluation(evals_result)]
    )
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
    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=1,
        valid_sets=lgb_eval,
        callbacks=[lgb.record_evaluation(evals_result)]
    )
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
    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=1,
        valid_sets=lgb_eval,
        callbacks=[lgb.record_evaluation(evals_result)]
    )
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
    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=1,
        valid_sets=lgb_eval,
        callbacks=[lgb.record_evaluation(evals_result)]
    )
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
    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=1,
        valid_sets=lgb_eval,
        callbacks=[lgb.record_evaluation(evals_result)]
    )
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
    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=1,
        valid_sets=lgb_eval,
        callbacks=[lgb.record_evaluation(evals_result)]
    )
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
    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=1,
        valid_sets=lgb_eval,
        callbacks=[lgb.record_evaluation(evals_result)]
    )
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
    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=50,
        valid_sets=lgb_eval,
        callbacks=[lgb.record_evaluation(evals_result)]
    )
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
    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=50,
        valid_sets=lgb_eval,
        callbacks=[lgb.record_evaluation(evals_result)]
    )
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

    pred_parameter["pred_early_stop_margin"] = 5.5
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
    est = lgb.train(
        dict(
            params,
            multi_error_top_k=1
        ),
        lgb_data,
        num_boost_round=10,
        valid_sets=[lgb_data],
        callbacks=[lgb.record_evaluation(results)]
    )
    predict_1 = est.predict(X)
    # check that default gives same result as k = 1
    np.testing.assert_allclose(predict_1, predict_default)
    # check against independent calculation for k = 1
    err = top_k_error(y, predict_1, 1)
    assert results['training']['multi_error'][-1] == pytest.approx(err)
    # check against independent calculation for k = 2
    results = {}
    est = lgb.train(
        dict(
            params,
            multi_error_top_k=2
        ),
        lgb_data,
        num_boost_round=10,
        valid_sets=[lgb_data],
        callbacks=[lgb.record_evaluation(results)]
    )
    predict_2 = est.predict(X)
    err = top_k_error(y, predict_2, 2)
    assert results['training']['multi_error@2'][-1] == pytest.approx(err)
    # check against independent calculation for k = 10
    results = {}
    est = lgb.train(
        dict(
            params,
            multi_error_top_k=10
        ),
        lgb_data,
        num_boost_round=10,
        valid_sets=[lgb_data],
        callbacks=[lgb.record_evaluation(results)]
    )
    predict_3 = est.predict(X)
    err = top_k_error(y, predict_3, 10)
    assert results['training']['multi_error@10'][-1] == pytest.approx(err)
    # check cases where predictions are equal
    X = np.array([[0, 0], [0, 0]])
    y = np.array([0, 1])
    lgb_data = lgb.Dataset(X, label=y)
    params['num_classes'] = 2
    results = {}
    lgb.train(
        params,
        lgb_data,
        num_boost_round=10,
        valid_sets=[lgb_data],
        callbacks=[lgb.record_evaluation(results)]
    )
    assert results['training']['multi_error'][-1] == pytest.approx(1)
    results = {}
    lgb.train(
        dict(
            params,
            multi_error_top_k=2
        ),
        lgb_data,
        num_boost_round=10,
        valid_sets=[lgb_data],
        callbacks=[lgb.record_evaluation(results)]
    )
    assert results['training']['multi_error@2'][-1] == pytest.approx(0)


@pytest.mark.skipif(getenv('TASK', '') == 'cuda', reason='Skip due to differences in implementation details of CUDA version')
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
    lgb.train(
        params,
        lgb_X,
        num_boost_round=10,
        valid_sets=[lgb_X],
        callbacks=[lgb.record_evaluation(results_auc_mu)]
    )
    params = {'objective': 'binary',
              'metric': 'auc',
              'verbose': -1,
              'seed': 0}
    results_auc = {}
    lgb.train(
        params,
        lgb_X,
        num_boost_round=10,
        valid_sets=[lgb_X],
        callbacks=[lgb.record_evaluation(results_auc)]
    )
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
    lgb.train(
        params,
        lgb_X,
        num_boost_round=10,
        valid_sets=[lgb_X],
        callbacks=[lgb.record_evaluation(results_auc_mu)]
    )
    assert results_auc_mu['training']['auc_mu'][-1] == pytest.approx(0.5)
    # test that weighted data gives different auc_mu
    lgb_X = lgb.Dataset(X, label=y)
    lgb_X_weighted = lgb.Dataset(X, label=y, weight=np.abs(np.random.normal(size=y.shape)))
    results_unweighted = {}
    results_weighted = {}
    params = dict(params, num_classes=10, num_leaves=5)
    lgb.train(
        params,
        lgb_X,
        num_boost_round=10,
        valid_sets=[lgb_X],
        callbacks=[lgb.record_evaluation(results_unweighted)]
    )
    lgb.train(
        params,
        lgb_X_weighted,
        num_boost_round=10,
        valid_sets=[lgb_X_weighted],
        callbacks=[lgb.record_evaluation(results_weighted)]
    )
    assert results_weighted['training']['auc_mu'][-1] < 1
    assert results_unweighted['training']['auc_mu'][-1] != results_weighted['training']['auc_mu'][-1]
    # test that equal data weights give same auc_mu as unweighted data
    lgb_X_weighted = lgb.Dataset(X, label=y, weight=np.ones(y.shape) * 0.5)
    lgb.train(
        params,
        lgb_X_weighted,
        num_boost_round=10,
        valid_sets=[lgb_X_weighted],
        callbacks=[lgb.record_evaluation(results_weighted)]
    )
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
    lgb.train(
        params,
        lgb_X,
        num_boost_round=100,
        valid_sets=[lgb_X],
        callbacks=[lgb.record_evaluation(results)]
    )
    assert results['training']['auc_mu'][-1] == pytest.approx(1)
    # test loading class weights
    Xy = np.loadtxt(
        str(Path(__file__).absolute().parents[2] / 'examples' / 'multiclass_classification' / 'multiclass.train')
    )
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
    lgb.train(
        params,
        lgb_X,
        num_boost_round=5,
        valid_sets=[lgb_X],
        callbacks=[lgb.record_evaluation(results_weight)]
    )
    params['auc_mu_weights'] = []
    results_no_weight = {}
    lgb.train(
        params,
        lgb_X,
        num_boost_round=5,
        valid_sets=[lgb_X],
        callbacks=[lgb.record_evaluation(results_no_weight)]
    )
    assert results_weight['training']['auc_mu'][-1] != results_no_weight['training']['auc_mu'][-1]


def test_ranking_prediction_early_stopping():
    rank_example_dir = Path(__file__).absolute().parents[2] / 'examples' / 'lambdarank'
    X_train, y_train = load_svmlight_file(str(rank_example_dir / 'rank.train'))
    q_train = np.loadtxt(str(rank_example_dir / 'rank.train.query'))
    X_test, _ = load_svmlight_file(str(rank_example_dir / 'rank.test'))
    params = {
        'objective': 'rank_xendcg',
        'verbose': -1
    }
    lgb_train = lgb.Dataset(X_train, y_train, group=q_train, params=params)
    gbm = lgb.train(params, lgb_train, num_boost_round=50)

    pred_parameter = {"pred_early_stop": True,
                      "pred_early_stop_freq": 5,
                      "pred_early_stop_margin": 1.5}
    ret_early = gbm.predict(X_test, **pred_parameter)

    pred_parameter["pred_early_stop_margin"] = 5.5
    ret_early_more_strict = gbm.predict(X_test, **pred_parameter)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(ret_early, ret_early_more_strict)


# Simulates position bias for a given ranking dataset.
# The ouput dataset is identical to the input one with the exception for the relevance labels.
# The new labels are generated according to an instance of a cascade user model:
# for each query, the user is simulated to be traversing the list of documents ranked by a baseline ranker
# (in our example it is simply the ordering by some feature correlated with relevance, e.g., 34)
# and clicks on that document (new_label=1) with some probability 'pclick' depending on its true relevance;
# at each position the user may stop the traversal with some probability pstop. For the non-clicked documents,
# new_label=0. Thus the generated new labels are biased towards the baseline ranker. 
# The positions of the documents in the ranked lists produced by the baseline, are returned.
def simulate_position_bias(file_dataset_in, file_query_in, file_dataset_out, baseline_feature):
    # a mapping of a document's true relevance (defined on a 5-grade scale) into the probability of clicking it
    def get_pclick(label):
        if label == 0:
            return 0.4
        elif label == 1:
            return 0.6
        elif label == 2:
            return 0.7
        elif label == 3:
            return 0.8
        else:
            return 0.9
    # an instantiation of a cascade model where the user stops with probability 0.2 after observing each document
    pstop = 0.2
 
    f_dataset_in = open(file_dataset_in, 'r')
    f_dataset_out = open(file_dataset_out, 'w')
    random.seed(10)
    positions_all = []
    for line in open(file_query_in):
        docs_num = int (line)
        lines = []
        index_values = []    
        positions = [0] * docs_num
        for index in range(docs_num):
            features = f_dataset_in.readline().split()
            lines.append(features)
            val = 0.0
            for feature_val in features:
                feature_val_split = feature_val.split(":")           
                if int(feature_val_split[0]) == baseline_feature:
                    val = float(feature_val_split[1])
            index_values.append([index, val])
        index_values.sort(key=lambda x: -x[1])
        stop = False 
        for pos in range(docs_num):
            index = index_values[pos][0]
            new_label = 0
            if not stop:
                label = int(lines[index][0])
                pclick = get_pclick(label)
                if random.random() < pclick:
                    new_label = 1       
                stop = random.random() < pstop
            lines[index][0] = str(new_label)
            positions[index] = pos
        for features in lines:
            f_dataset_out.write(' '.join(features) + '\n')
        positions_all.extend(positions)
    f_dataset_out.close()
    return positions_all


@pytest.mark.skipif(getenv('TASK', '') == 'cuda', reason='Positions in learning to rank is not supported in CUDA version yet')
def test_ranking_with_position_information_with_file(tmp_path):
    rank_example_dir = Path(__file__).absolute().parents[2] / 'examples' / 'lambdarank'
    params = {
        'objective': 'lambdarank',
        'verbose': -1,
        'eval_at': [3],
        'metric': 'ndcg',
        'bagging_freq': 1,
        'bagging_fraction': 0.9,
        'min_data_in_leaf': 50,
        'min_sum_hessian_in_leaf': 5.0
    }

    # simulate position bias for the train dataset and put the train dataset with biased labels to temp directory
    positions = simulate_position_bias(str(rank_example_dir / 'rank.train'), str(rank_example_dir / 'rank.train.query'), str(tmp_path / 'rank.train'), baseline_feature=34)
    copyfile(str(rank_example_dir / 'rank.train.query'), str(tmp_path / 'rank.train.query'))
    copyfile(str(rank_example_dir / 'rank.test'), str(tmp_path / 'rank.test'))
    copyfile(str(rank_example_dir / 'rank.test.query'), str(tmp_path / 'rank.test.query'))

    lgb_train = lgb.Dataset(str(tmp_path / 'rank.train'), params=params)
    lgb_valid = [lgb_train.create_valid(str(tmp_path / 'rank.test'))]
    gbm_baseline = lgb.train(params, lgb_train, valid_sets = lgb_valid, num_boost_round=50)

    f_positions_out = open(str(tmp_path / 'rank.train.position'), 'w')
    for pos in positions:
        f_positions_out.write(str(pos) + '\n')
    f_positions_out.close()

    lgb_train = lgb.Dataset(str(tmp_path / 'rank.train'), params=params)
    lgb_valid = [lgb_train.create_valid(str(tmp_path / 'rank.test'))]
    gbm_unbiased_with_file = lgb.train(params, lgb_train, valid_sets = lgb_valid, num_boost_round=50)
    
    # the performance of the unbiased LambdaMART should outperform the plain LambdaMART on the dataset with position bias
    assert gbm_baseline.best_score['valid_0']['ndcg@3'] + 0.03 <= gbm_unbiased_with_file.best_score['valid_0']['ndcg@3']

    # add extra row to position file
    with open(str(tmp_path / 'rank.train.position'), 'a') as file:
        file.write('pos_1000\n')
        file.close()
    lgb_train = lgb.Dataset(str(tmp_path / 'rank.train'), params=params)
    lgb_valid = [lgb_train.create_valid(str(tmp_path / 'rank.test'))]
    with pytest.raises(lgb.basic.LightGBMError, match="Positions size \(3006\) doesn't match data size"):
        lgb.train(params, lgb_train, valid_sets = lgb_valid, num_boost_round=50)


@pytest.mark.skipif(getenv('TASK', '') == 'cuda', reason='Positions in learning to rank is not supported in CUDA version yet')
def test_ranking_with_position_information_with_dataset_constructor(tmp_path):
    rank_example_dir = Path(__file__).absolute().parents[2] / 'examples' / 'lambdarank'
    params = {
        'objective': 'lambdarank',
        'verbose': -1,
        'eval_at': [3],
        'metric': 'ndcg',
        'bagging_freq': 1,
        'bagging_fraction': 0.9,
        'min_data_in_leaf': 50,
        'min_sum_hessian_in_leaf': 5.0,
        'num_threads': 1,
        'deterministic': True,
        'seed': 0
    }

    # simulate position bias for the train dataset and put the train dataset with biased labels to temp directory
    positions = simulate_position_bias(str(rank_example_dir / 'rank.train'), str(rank_example_dir / 'rank.train.query'), str(tmp_path / 'rank.train'), baseline_feature=34)
    copyfile(str(rank_example_dir / 'rank.train.query'), str(tmp_path / 'rank.train.query'))
    copyfile(str(rank_example_dir / 'rank.test'), str(tmp_path / 'rank.test'))
    copyfile(str(rank_example_dir / 'rank.test.query'), str(tmp_path / 'rank.test.query'))

    lgb_train = lgb.Dataset(str(tmp_path / 'rank.train'), params=params)
    lgb_valid = [lgb_train.create_valid(str(tmp_path / 'rank.test'))]
    gbm_baseline = lgb.train(params, lgb_train, valid_sets = lgb_valid, num_boost_round=50)

    positions = np.array(positions)

    # test setting positions through Dataset constructor with numpy array
    lgb_train = lgb.Dataset(str(tmp_path / 'rank.train'), params=params, position=positions)
    lgb_valid = [lgb_train.create_valid(str(tmp_path / 'rank.test'))]
    gbm_unbiased = lgb.train(params, lgb_train, valid_sets = lgb_valid, num_boost_round=50)

    # the performance of the unbiased LambdaMART should outperform the plain LambdaMART on the dataset with position bias
    assert gbm_baseline.best_score['valid_0']['ndcg@3'] + 0.03 <= gbm_unbiased.best_score['valid_0']['ndcg@3']

    if PANDAS_INSTALLED:
        # test setting positions through Dataset constructor with pandas Series
        lgb_train = lgb.Dataset(str(tmp_path / 'rank.train'), params=params, position=pd_Series(positions))
        lgb_valid = [lgb_train.create_valid(str(tmp_path / 'rank.test'))]
        gbm_unbiased_pandas_series = lgb.train(params, lgb_train, valid_sets = lgb_valid, num_boost_round=50)
        assert gbm_unbiased.best_score['valid_0']['ndcg@3'] == gbm_unbiased_pandas_series.best_score['valid_0']['ndcg@3']

    # test setting positions through set_position
    lgb_train = lgb.Dataset(str(tmp_path / 'rank.train'), params=params)
    lgb_valid = [lgb_train.create_valid(str(tmp_path / 'rank.test'))]
    lgb_train.set_position(positions)
    gbm_unbiased_set_position = lgb.train(params, lgb_train, valid_sets = lgb_valid, num_boost_round=50)
    assert gbm_unbiased.best_score['valid_0']['ndcg@3'] == gbm_unbiased_set_position.best_score['valid_0']['ndcg@3']

    # test get_position works
    positions_from_get = lgb_train.get_position()
    np.testing.assert_array_equal(positions_from_get, positions)


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
                    callbacks=[lgb.early_stopping(stopping_rounds=5)])
    assert gbm.best_iteration == 10
    assert valid_set_name in gbm.best_score
    assert 'binary_logloss' in gbm.best_score[valid_set_name]
    # early stopping occurs
    gbm = lgb.train(params, lgb_train,
                    num_boost_round=40,
                    valid_sets=lgb_eval,
                    valid_names=valid_set_name,
                    callbacks=[lgb.early_stopping(stopping_rounds=5)])
    assert gbm.best_iteration <= 39
    assert valid_set_name in gbm.best_score
    assert 'binary_logloss' in gbm.best_score[valid_set_name]


@pytest.mark.parametrize('use_valid', [True, False])
def test_early_stopping_ignores_training_set(use_valid):
    x = np.linspace(-1, 1, 100)
    X = x.reshape(-1, 1)
    y = x**2
    X_train, X_valid = X[:80], X[80:]
    y_train, y_valid = y[:80], y[80:]
    train_ds = lgb.Dataset(X_train, y_train)
    valid_ds = lgb.Dataset(X_valid, y_valid)
    valid_sets = [train_ds]
    valid_names = ['train']
    if use_valid:
        valid_sets.append(valid_ds)
        valid_names.append('valid')
    eval_result = {}

    def train_fn():
        return lgb.train(
            {'num_leaves': 5},
            train_ds,
            num_boost_round=2,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[lgb.early_stopping(1), lgb.record_evaluation(eval_result)]
        )
    if use_valid:
        bst = train_fn()
        assert bst.best_iteration == 1
        assert eval_result['train']['l2'][1] < eval_result['train']['l2'][0]  # train improved
        assert eval_result['valid']['l2'][1] > eval_result['valid']['l2'][0]  # valid didn't
    else:
        with pytest.warns(UserWarning, match='Only training set found, disabling early stopping.'):
            bst = train_fn()
        assert bst.current_iteration() == 2
        assert bst.best_iteration == 0


@pytest.mark.parametrize('first_metric_only', [True, False])
def test_early_stopping_via_global_params(first_metric_only):
    X, y = load_breast_cancer(return_X_y=True)
    num_trees = 5
    params = {
        'num_trees': num_trees,
        'objective': 'binary',
        'metric': 'None',
        'verbose': -1,
        'early_stopping_round': 2,
        'first_metric_only': first_metric_only
    }
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    valid_set_name = 'valid_set'
    gbm = lgb.train(params,
                    lgb_train,
                    feval=[decreasing_metric, constant_metric],
                    valid_sets=lgb_eval,
                    valid_names=valid_set_name)
    if first_metric_only:
        assert gbm.best_iteration == num_trees
    else:
        assert gbm.best_iteration == 1
    assert valid_set_name in gbm.best_score
    assert 'decreasing_metric' in gbm.best_score[valid_set_name]
    assert 'error' in gbm.best_score[valid_set_name]


@pytest.mark.parametrize('first_only', [True, False])
@pytest.mark.parametrize('single_metric', [True, False])
@pytest.mark.parametrize('greater_is_better', [True, False])
def test_early_stopping_min_delta(first_only, single_metric, greater_is_better):
    if single_metric and not first_only:
        pytest.skip("first_metric_only doesn't affect single metric.")
    metric2min_delta = {
        'auc': 0.001,
        'binary_logloss': 0.01,
        'average_precision': 0.001,
        'mape': 0.01,
    }
    if single_metric:
        if greater_is_better:
            metric = 'auc'
        else:
            metric = 'binary_logloss'
    else:
        if first_only:
            if greater_is_better:
                metric = ['auc', 'binary_logloss']
            else:
                metric = ['binary_logloss', 'auc']
        else:
            if greater_is_better:
                metric = ['auc', 'average_precision']
            else:
                metric = ['binary_logloss', 'mape']

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)
    train_ds = lgb.Dataset(X_train, y_train)
    valid_ds = lgb.Dataset(X_valid, y_valid, reference=train_ds)

    params = {'objective': 'binary', 'metric': metric, 'verbose': -1}
    if isinstance(metric, str):
        min_delta = metric2min_delta[metric]
    elif first_only:
        min_delta = metric2min_delta[metric[0]]
    else:
        min_delta = [metric2min_delta[m] for m in metric]
    train_kwargs = {
        "params": params,
        "train_set": train_ds,
        "num_boost_round": 50,
        "valid_sets": [train_ds, valid_ds],
        "valid_names": ['training', 'valid'],
    }

    # regular early stopping
    evals_result = {}
    train_kwargs['callbacks'] = [
        lgb.callback.early_stopping(10, first_only, verbose=False),
        lgb.record_evaluation(evals_result)
    ]
    bst = lgb.train(**train_kwargs)
    scores = np.vstack(list(evals_result['valid'].values())).T

    # positive min_delta
    delta_result = {}
    train_kwargs['callbacks'] = [
        lgb.callback.early_stopping(10, first_only, verbose=False, min_delta=min_delta),
        lgb.record_evaluation(delta_result)
    ]
    delta_bst = lgb.train(**train_kwargs)
    delta_scores = np.vstack(list(delta_result['valid'].values())).T

    if first_only:
        scores = scores[:, 0]
        delta_scores = delta_scores[:, 0]

    assert delta_bst.num_trees() < bst.num_trees()
    np.testing.assert_allclose(scores[:len(delta_scores)], delta_scores)
    last_score = delta_scores[-1]
    best_score = delta_scores[delta_bst.num_trees() - 1]
    if greater_is_better:
        assert np.less_equal(last_score, best_score + min_delta).any()
    else:
        assert np.greater_equal(last_score, best_score - min_delta).any()


def test_continue_train():
    X, y = make_synthetic_regression()
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
    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=30,
        valid_sets=lgb_eval,
        # test custom eval metrics
        feval=(lambda p, d: ('custom_mae', mean_absolute_error(p, d.get_label()), False)),
        callbacks=[lgb.record_evaluation(evals_result)],
        init_model='model.txt'
    )
    ret = mean_absolute_error(y_test, gbm.predict(X_test))
    assert ret < 13.6
    assert evals_result['valid_0']['l1'][-1] == pytest.approx(ret)
    np.testing.assert_allclose(evals_result['valid_0']['l1'], evals_result['valid_0']['custom_mae'])


def test_continue_train_reused_dataset():
    X, y = make_synthetic_regression()
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
    X, y = make_synthetic_regression()
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
    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=50,
        valid_sets=lgb_eval,
        callbacks=[lgb.record_evaluation(evals_result)],
        init_model=init_gbm
    )
    ret = mean_absolute_error(y_test, gbm.predict(X_test))
    assert ret < 13.6
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
    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=30,
        valid_sets=lgb_eval,
        callbacks=[lgb.record_evaluation(evals_result)],
        init_model=init_gbm
    )
    ret = multi_logloss(y_test, gbm.predict(X_test))
    assert ret < 0.1
    assert evals_result['valid_0']['multi_logloss'][-1] == pytest.approx(ret)


def test_cv():
    X_train, y_train = make_synthetic_regression()
    params = {'verbose': -1}
    lgb_train = lgb.Dataset(X_train, y_train)
    # shuffle = False, override metric in params
    params_with_metric = {'metric': 'l2', 'verbose': -1}
    cv_res = lgb.cv(params_with_metric, lgb_train, num_boost_round=10,
                    nfold=3, stratified=False, shuffle=False, metrics='l1')
    assert 'valid l1-mean' in cv_res
    assert 'valid l2-mean' not in cv_res
    assert len(cv_res['valid l1-mean']) == 10
    # shuffle = True, callbacks
    cv_res = lgb.cv(params, lgb_train, num_boost_round=10, nfold=3,
                    stratified=False, shuffle=True, metrics='l1',
                    callbacks=[lgb.reset_parameter(learning_rate=lambda i: 0.1 - 0.001 * i)])
    assert 'valid l1-mean' in cv_res
    assert len(cv_res['valid l1-mean']) == 10
    # enable display training loss
    cv_res = lgb.cv(params_with_metric, lgb_train, num_boost_round=10,
                    nfold=3, stratified=False, shuffle=False,
                    metrics='l1', eval_train_metric=True)
    assert 'train l1-mean' in cv_res
    assert 'valid l1-mean' in cv_res
    assert 'train l2-mean' not in cv_res
    assert 'valid l2-mean' not in cv_res
    assert len(cv_res['train l1-mean']) == 10
    assert len(cv_res['valid l1-mean']) == 10
    # self defined folds
    tss = TimeSeriesSplit(3)
    folds = tss.split(X_train)
    cv_res_gen = lgb.cv(params_with_metric, lgb_train, num_boost_round=10, folds=folds)
    cv_res_obj = lgb.cv(params_with_metric, lgb_train, num_boost_round=10, folds=tss)
    np.testing.assert_allclose(cv_res_gen['valid l2-mean'], cv_res_obj['valid l2-mean'])
    # LambdaRank
    rank_example_dir = Path(__file__).absolute().parents[2] / 'examples' / 'lambdarank'
    X_train, y_train = load_svmlight_file(str(rank_example_dir / 'rank.train'))
    q_train = np.loadtxt(str(rank_example_dir / 'rank.train.query'))
    params_lambdarank = {'objective': 'lambdarank', 'verbose': -1, 'eval_at': 3}
    lgb_train = lgb.Dataset(X_train, y_train, group=q_train)
    # ... with l2 metric
    cv_res_lambda = lgb.cv(params_lambdarank, lgb_train, num_boost_round=10, nfold=3, metrics='l2')
    assert len(cv_res_lambda) == 2
    assert not np.isnan(cv_res_lambda['valid l2-mean']).any()
    # ... with NDCG (default) metric
    cv_res_lambda = lgb.cv(params_lambdarank, lgb_train, num_boost_round=10, nfold=3)
    assert len(cv_res_lambda) == 2
    assert not np.isnan(cv_res_lambda['valid ndcg@3-mean']).any()
    # self defined folds with lambdarank
    cv_res_lambda_obj = lgb.cv(params_lambdarank, lgb_train, num_boost_round=10,
                               folds=GroupKFold(n_splits=3))
    np.testing.assert_allclose(cv_res_lambda['valid ndcg@3-mean'], cv_res_lambda_obj['valid ndcg@3-mean'])


def test_cv_works_with_init_model(tmp_path):
    X, y = make_synthetic_regression()
    params = {'objective': 'regression', 'verbose': -1}
    num_train_rounds = 2
    lgb_train = lgb.Dataset(X, y, free_raw_data=False)
    bst = lgb.train(
        params=params,
        train_set=lgb_train,
        num_boost_round=num_train_rounds
    )
    preds_raw = bst.predict(X, raw_score=True)
    model_path_txt = str(tmp_path / 'lgb.model')
    bst.save_model(model_path_txt)

    num_cv_rounds = 5
    cv_kwargs = {
        "num_boost_round": num_cv_rounds,
        "nfold": 3,
        "stratified": False,
        "shuffle": False,
        "seed": 708,
        "return_cvbooster": True,
        "params": params
    }

    # init_model from an in-memory Booster
    cv_res = lgb.cv(
        train_set=lgb_train,
        init_model=bst,
        **cv_kwargs
    )
    cv_bst_w_in_mem_init_model = cv_res["cvbooster"]
    assert cv_bst_w_in_mem_init_model.current_iteration() == [num_train_rounds + num_cv_rounds] * 3
    for booster in cv_bst_w_in_mem_init_model.boosters:
        np.testing.assert_allclose(
            preds_raw,
            booster.predict(X, raw_score=True, num_iteration=num_train_rounds)
        )

    # init_model from a text file
    cv_res = lgb.cv(
        train_set=lgb_train,
        init_model=model_path_txt,
        **cv_kwargs
    )
    cv_bst_w_file_init_model = cv_res["cvbooster"]
    assert cv_bst_w_file_init_model.current_iteration() == [num_train_rounds + num_cv_rounds] * 3
    for booster in cv_bst_w_file_init_model.boosters:
        np.testing.assert_allclose(
            preds_raw,
            booster.predict(X, raw_score=True, num_iteration=num_train_rounds)
        )

    # predictions should be identical
    for i in range(3):
        np.testing.assert_allclose(
            cv_bst_w_in_mem_init_model.boosters[i].predict(X),
            cv_bst_w_file_init_model.boosters[i].predict(X)
        )


def test_cvbooster():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbose': -1,
    }
    nfold = 3
    lgb_train = lgb.Dataset(X_train, y_train)
    # with early stopping
    cv_res = lgb.cv(params, lgb_train,
                    num_boost_round=25,
                    nfold=nfold,
                    callbacks=[lgb.early_stopping(stopping_rounds=5)],
                    return_cvbooster=True)
    assert 'cvbooster' in cv_res
    cvb = cv_res['cvbooster']
    assert isinstance(cvb, lgb.CVBooster)
    assert isinstance(cvb.boosters, list)
    assert len(cvb.boosters) == nfold
    assert all(isinstance(bst, lgb.Booster) for bst in cvb.boosters)
    assert cvb.best_iteration > 0
    # predict by each fold booster
    preds = cvb.predict(X_test)
    assert isinstance(preds, list)
    assert len(preds) == nfold
    # check that each booster predicted using the best iteration
    for fold_preds, bst in zip(preds, cvb.boosters):
        assert bst.best_iteration == cvb.best_iteration
        expected = bst.predict(X_test, num_iteration=cvb.best_iteration)
        np.testing.assert_allclose(fold_preds, expected)
    # fold averaging
    avg_pred = np.mean(preds, axis=0)
    ret = log_loss(y_test, avg_pred)
    assert ret < 0.13
    # without early stopping
    cv_res = lgb.cv(params, lgb_train,
                    num_boost_round=20,
                    nfold=3,
                    return_cvbooster=True)
    cvb = cv_res['cvbooster']
    assert cvb.best_iteration == -1
    preds = cvb.predict(X_test)
    avg_pred = np.mean(preds, axis=0)
    ret = log_loss(y_test, avg_pred)
    assert ret < 0.15


def test_cvbooster_save_load(tmp_path):
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.1, random_state=42)
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbose': -1,
    }
    nfold = 3
    lgb_train = lgb.Dataset(X_train, y_train)

    cv_res = lgb.cv(params, lgb_train,
                    num_boost_round=10,
                    nfold=nfold,
                    callbacks=[lgb.early_stopping(stopping_rounds=5)],
                    return_cvbooster=True)
    cvbooster = cv_res['cvbooster']
    preds = cvbooster.predict(X_test)
    best_iteration = cvbooster.best_iteration

    model_path_txt = str(tmp_path / 'lgb.model')

    cvbooster.save_model(model_path_txt)
    model_string = cvbooster.model_to_string()
    del cvbooster

    cvbooster_from_txt_file = lgb.CVBooster(model_file=model_path_txt)
    cvbooster_from_string = lgb.CVBooster().model_from_string(model_string)
    for cvbooster_loaded in [cvbooster_from_txt_file, cvbooster_from_string]:
        assert best_iteration == cvbooster_loaded.best_iteration
        np.testing.assert_array_equal(preds, cvbooster_loaded.predict(X_test))


@pytest.mark.parametrize('serializer', SERIALIZERS)
def test_cvbooster_picklable(serializer):
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.1, random_state=42)
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbose': -1,
    }
    nfold = 3
    lgb_train = lgb.Dataset(X_train, y_train)

    cv_res = lgb.cv(params, lgb_train,
                    num_boost_round=10,
                    nfold=nfold,
                    callbacks=[lgb.early_stopping(stopping_rounds=5)],
                    return_cvbooster=True)
    cvbooster = cv_res['cvbooster']
    preds = cvbooster.predict(X_test)
    best_iteration = cvbooster.best_iteration

    cvbooster_from_disk = pickle_and_unpickle_object(obj=cvbooster, serializer=serializer)
    del cvbooster

    assert best_iteration == cvbooster_from_disk.best_iteration

    preds_from_disk = cvbooster_from_disk.predict(X_test)
    np.testing.assert_array_equal(preds, preds_from_disk)


def test_feature_name():
    X_train, y_train = make_synthetic_regression()
    params = {'verbose': -1}
    lgb_train = lgb.Dataset(X_train, y_train)
    feature_names = [f'f_{i}' for i in range(X_train.shape[-1])]
    gbm = lgb.train(params, lgb_train, num_boost_round=5, feature_name=feature_names)
    assert feature_names == gbm.feature_name()
    # test feature_names with whitespaces
    feature_names_with_space = [f'f {i}' for i in range(X_train.shape[-1])]
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


def test_parameters_are_loaded_from_model_file(tmp_path):
    X = np.hstack([np.random.rand(100, 1), np.random.randint(0, 5, (100, 2))])
    y = np.random.rand(100)
    ds = lgb.Dataset(X, y)
    params = {
        'bagging_fraction': 0.8,
        'bagging_freq': 2,
        'boosting': 'rf',
        'feature_contri': [0.5, 0.5, 0.5],
        'feature_fraction': 0.7,
        'boost_from_average': False,
        'interaction_constraints': [[0, 1], [0]],
        'metric': ['l2', 'rmse'],
        'num_leaves': 5,
        'num_threads': 1,
    }
    model_file = tmp_path / 'model.txt'
    lgb.train(params, ds, num_boost_round=1, categorical_feature=[1, 2]).save_model(model_file)
    bst = lgb.Booster(model_file=model_file)
    set_params = {k: bst.params[k] for k in params.keys()}
    assert set_params == params
    assert bst.params['categorical_feature'] == [1, 2]

    # check that passing parameters to the constructor raises warning and ignores them
    with pytest.warns(UserWarning, match='Ignoring params argument'):
        bst2 = lgb.Booster(params={'num_leaves': 7}, model_file=model_file)
    assert bst.params == bst2.params


def test_save_load_copy_pickle():
    def train_and_predict(init_model=None, return_model=False):
        X, y = make_synthetic_regression()
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
    gbm4.model_from_string(model_str)
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
    X = pd.DataFrame({"A": pd.arrays.SparseArray(np.random.permutation([0, 1, 2] * 100)),
                      "B": pd.arrays.SparseArray(np.random.permutation([0.0, 0.1, 0.2, -0.1, 0.2] * 60)),
                      "C": pd.arrays.SparseArray(np.random.permutation([True, False] * 150))})
    y = pd.Series(pd.arrays.SparseArray(np.random.permutation([0, 1] * 150)))
    X_test = pd.DataFrame({"A": pd.arrays.SparseArray(np.random.permutation([0, 2] * 30)),
                           "B": pd.arrays.SparseArray(np.random.permutation([0.0, 0.1, 0.2, -0.1] * 15)),
                           "C": pd.arrays.SparseArray(np.random.permutation([True, False] * 30))})
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
    lgb.train(
        params,
        tmp_dat_train,
        num_boost_round=20,
        valid_sets=[tmp_dat_train, tmp_dat_val],
        callbacks=[lgb.record_evaluation(evals_result)]
    )
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
    contribs_csr_array = np.swapaxes(np.array([sparse_array.toarray() for sparse_array in contribs_csr]), 0, 1)
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
    contribs_csc_array = np.swapaxes(np.array([sparse_array.toarray() for sparse_array in contribs_csc]), 0, 1)
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


def test_training_on_constructed_subset_without_params():
    X = np.random.random((100, 10))
    y = np.random.random(100)
    lgb_data = lgb.Dataset(X, y)
    subset_indices = [1, 2, 3, 4]
    subset = lgb_data.subset(subset_indices).construct()
    bst = lgb.train({}, subset, num_boost_round=1)
    assert subset.get_params() == {}
    assert subset.num_data() == len(subset_indices)
    assert bst.current_iteration() == 1


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
    return lgb.Dataset(x, label=y, categorical_feature=categorical_features, free_raw_data=False)


@pytest.mark.skipif(getenv('TASK', '') == 'cuda', reason='Monotone constraints are not yet supported by CUDA version')
@pytest.mark.parametrize("test_with_categorical_variable", [True, False])
def test_monotone_constraints(test_with_categorical_variable):
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
                features = {f"Column_{f}" for f in features}
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

    trainset = generate_trainset_for_monotone_constraints_tests(
        test_with_categorical_variable
    )
    for test_with_interaction_constraints in [True, False]:
        error_msg = ("Model not correctly constrained "
                     f"(test_with_interaction_constraints={test_with_interaction_constraints})")
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
            ), error_msg
            if test_with_interaction_constraints:
                feature_sets = [["Column_0"], ["Column_1"], "Column_2"]
                assert are_interactions_enforced(constrained_model, feature_sets)


@pytest.mark.skipif(getenv('TASK', '') == 'cuda', reason='Monotone constraints are not yet supported by CUDA version')
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
@pytest.mark.skipif(getenv('TASK', '') == 'cuda', reason='Monotone constraints are not yet supported by CUDA version')
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
    unconstrained_model_predictions = unconstrained_model.predict(
        x3_negatively_correlated_with_y.reshape(-1, 1)
    )

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
    x = np.ones((100, 1))
    x[:30, 0] = -1
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


def test_refit_dataset_params():
    # check refit accepts dataset_params
    X, y = load_breast_cancer(return_X_y=True)
    lgb_train = lgb.Dataset(X, y, init_score=np.zeros(y.size))
    train_params = {
        'objective': 'binary',
        'verbose': -1,
        'seed': 123
    }
    gbm = lgb.train(train_params, lgb_train, num_boost_round=10)
    non_weight_err_pred = log_loss(y, gbm.predict(X))
    refit_weight = np.random.rand(y.shape[0])
    dataset_params = {
        'max_bin': 260,
        'min_data_in_bin': 5,
        'data_random_seed': 123,
    }
    new_gbm = gbm.refit(
        data=X,
        label=y,
        weight=refit_weight,
        dataset_params=dataset_params,
        decay_rate=0.0,
    )
    weight_err_pred = log_loss(y, new_gbm.predict(X))
    train_set_params = new_gbm.train_set.get_params()
    stored_weights = new_gbm.train_set.get_weight()
    assert weight_err_pred != non_weight_err_pred
    assert train_set_params["max_bin"] == 260
    assert train_set_params["min_data_in_bin"] == 5
    assert train_set_params["data_random_seed"] == 123
    np.testing.assert_allclose(stored_weights, refit_weight)


@pytest.mark.parametrize('boosting_type', ['rf', 'dart'])
def test_mape_for_specific_boosting_types(boosting_type):
    X, y = make_synthetic_regression()
    y = abs(y)
    params = {
        'boosting_type': boosting_type,
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
    # the following checks that dart and rf with mape can predict outside the 0-1 range
    # https://github.com/microsoft/LightGBM/issues/1579
    assert pred_mean > 8


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
    assert 'valid multi_logloss-mean' in results
    assert len(results['valid multi_logloss-mean']) == 10


def test_metrics():
    X, y = load_digits(n_class=2, return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_valid = lgb.Dataset(X_test, y_test, reference=lgb_train)

    evals_result = {}
    params_dummy_obj_verbose = {'verbose': -1, 'objective': dummy_obj}
    params_obj_verbose = {'objective': 'binary', 'verbose': -1}
    params_obj_metric_log_verbose = {'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1}
    params_obj_metric_err_verbose = {'objective': 'binary', 'metric': 'binary_error', 'verbose': -1}
    params_obj_metric_inv_verbose = {'objective': 'binary', 'metric': 'invalid_metric', 'verbose': -1}
    params_obj_metric_quant_verbose = {'objective': 'regression', 'metric': 'quantile', 'verbose': 2}
    params_obj_metric_multi_verbose = {'objective': 'binary',
                                       'metric': ['binary_logloss', 'binary_error'],
                                       'verbose': -1}
    params_obj_metric_none_verbose = {'objective': 'binary', 'metric': 'None', 'verbose': -1}
    params_dummy_obj_metric_log_verbose = {'objective': dummy_obj, 'metric': 'binary_logloss', 'verbose': -1}
    params_dummy_obj_metric_err_verbose = {'objective': dummy_obj, 'metric': 'binary_error', 'verbose': -1}
    params_dummy_obj_metric_inv_verbose = {'objective': dummy_obj, 'metric_types': 'invalid_metric', 'verbose': -1}
    params_dummy_obj_metric_multi_verbose = {'objective': dummy_obj, 'metric': ['binary_logloss', 'binary_error'], 'verbose': -1}
    params_dummy_obj_metric_none_verbose = {'objective': dummy_obj, 'metric': 'None', 'verbose': -1}

    def get_cv_result(params=params_obj_verbose, **kwargs):
        return lgb.cv(params, lgb_train, num_boost_round=2, **kwargs)

    def train_booster(params=params_obj_verbose, **kwargs):
        lgb.train(
            params,
            lgb_train,
            num_boost_round=2,
            valid_sets=[lgb_valid],
            callbacks=[lgb.record_evaluation(evals_result)],
            **kwargs
        )

    # no custom objective, no feval
    # default metric
    res = get_cv_result()
    assert len(res) == 2
    assert 'valid binary_logloss-mean' in res

    # non-default metric in params
    res = get_cv_result(params=params_obj_metric_err_verbose)
    assert len(res) == 2
    assert 'valid binary_error-mean' in res

    # default metric in args
    res = get_cv_result(metrics='binary_logloss')
    assert len(res) == 2
    assert 'valid binary_logloss-mean' in res

    # non-default metric in args
    res = get_cv_result(metrics='binary_error')
    assert len(res) == 2
    assert 'valid binary_error-mean' in res

    # metric in args overwrites one in params
    res = get_cv_result(params=params_obj_metric_inv_verbose, metrics='binary_error')
    assert len(res) == 2
    assert 'valid binary_error-mean' in res

    # metric in args overwrites one in params
    res = get_cv_result(params=params_obj_metric_quant_verbose)
    assert len(res) == 2
    assert 'valid quantile-mean' in res

    # multiple metrics in params
    res = get_cv_result(params=params_obj_metric_multi_verbose)
    assert len(res) == 4
    assert 'valid binary_logloss-mean' in res
    assert 'valid binary_error-mean' in res

    # multiple metrics in args
    res = get_cv_result(metrics=['binary_logloss', 'binary_error'])
    assert len(res) == 4
    assert 'valid binary_logloss-mean' in res
    assert 'valid binary_error-mean' in res

    # remove default metric by 'None' in list
    res = get_cv_result(metrics=['None'])
    assert len(res) == 0

    # remove default metric by 'None' aliases
    for na_alias in ('None', 'na', 'null', 'custom'):
        res = get_cv_result(metrics=na_alias)
        assert len(res) == 0

    # custom objective, no feval
    # no default metric
    res = get_cv_result(params=params_dummy_obj_verbose)
    assert len(res) == 0

    # metric in params
    res = get_cv_result(params=params_dummy_obj_metric_err_verbose)
    assert len(res) == 2
    assert 'valid binary_error-mean' in res

    # metric in args
    res = get_cv_result(params=params_dummy_obj_verbose, metrics='binary_error')
    assert len(res) == 2
    assert 'valid binary_error-mean' in res

    # metric in args overwrites its' alias in params
    res = get_cv_result(params=params_dummy_obj_metric_inv_verbose, metrics='binary_error')
    assert len(res) == 2
    assert 'valid binary_error-mean' in res

    # multiple metrics in params
    res = get_cv_result(params=params_dummy_obj_metric_multi_verbose)
    assert len(res) == 4
    assert 'valid binary_logloss-mean' in res
    assert 'valid binary_error-mean' in res

    # multiple metrics in args
    res = get_cv_result(params=params_dummy_obj_verbose,
                        metrics=['binary_logloss', 'binary_error'])
    assert len(res) == 4
    assert 'valid binary_logloss-mean' in res
    assert 'valid binary_error-mean' in res

    # no custom objective, feval
    # default metric with custom one
    res = get_cv_result(feval=constant_metric)
    assert len(res) == 4
    assert 'valid binary_logloss-mean' in res
    assert 'valid error-mean' in res

    # non-default metric in params with custom one
    res = get_cv_result(params=params_obj_metric_err_verbose, feval=constant_metric)
    assert len(res) == 4
    assert 'valid binary_error-mean' in res
    assert 'valid error-mean' in res

    # default metric in args with custom one
    res = get_cv_result(metrics='binary_logloss', feval=constant_metric)
    assert len(res) == 4
    assert 'valid binary_logloss-mean' in res
    assert 'valid error-mean' in res

    # non-default metric in args with custom one
    res = get_cv_result(metrics='binary_error', feval=constant_metric)
    assert len(res) == 4
    assert 'valid binary_error-mean' in res
    assert 'valid error-mean' in res

    # metric in args overwrites one in params, custom one is evaluated too
    res = get_cv_result(params=params_obj_metric_inv_verbose, metrics='binary_error', feval=constant_metric)
    assert len(res) == 4
    assert 'valid binary_error-mean' in res
    assert 'valid error-mean' in res

    # multiple metrics in params with custom one
    res = get_cv_result(params=params_obj_metric_multi_verbose, feval=constant_metric)
    assert len(res) == 6
    assert 'valid binary_logloss-mean' in res
    assert 'valid binary_error-mean' in res
    assert 'valid error-mean' in res

    # multiple metrics in args with custom one
    res = get_cv_result(metrics=['binary_logloss', 'binary_error'], feval=constant_metric)
    assert len(res) == 6
    assert 'valid binary_logloss-mean' in res
    assert 'valid binary_error-mean' in res
    assert 'valid error-mean' in res

    # custom metric is evaluated despite 'None' is passed
    res = get_cv_result(metrics=['None'], feval=constant_metric)
    assert len(res) == 2
    assert 'valid error-mean' in res

    # custom objective, feval
    # no default metric, only custom one
    res = get_cv_result(params=params_dummy_obj_verbose, feval=constant_metric)
    assert len(res) == 2
    assert 'valid error-mean' in res

    # metric in params with custom one
    res = get_cv_result(params=params_dummy_obj_metric_err_verbose, feval=constant_metric)
    assert len(res) == 4
    assert 'valid binary_error-mean' in res
    assert 'valid error-mean' in res

    # metric in args with custom one
    res = get_cv_result(params=params_dummy_obj_verbose,
                        feval=constant_metric, metrics='binary_error')
    assert len(res) == 4
    assert 'valid binary_error-mean' in res
    assert 'valid error-mean' in res

    # metric in args overwrites one in params, custom one is evaluated too
    res = get_cv_result(params=params_dummy_obj_metric_inv_verbose,
                        feval=constant_metric, metrics='binary_error')
    assert len(res) == 4
    assert 'valid binary_error-mean' in res
    assert 'valid error-mean' in res

    # multiple metrics in params with custom one
    res = get_cv_result(params=params_dummy_obj_metric_multi_verbose, feval=constant_metric)
    assert len(res) == 6
    assert 'valid binary_logloss-mean' in res
    assert 'valid binary_error-mean' in res
    assert 'valid error-mean' in res

    # multiple metrics in args with custom one
    res = get_cv_result(params=params_dummy_obj_verbose, feval=constant_metric,
                        metrics=['binary_logloss', 'binary_error'])
    assert len(res) == 6
    assert 'valid binary_logloss-mean' in res
    assert 'valid binary_error-mean' in res
    assert 'valid error-mean' in res

    # custom metric is evaluated despite 'None' is passed
    res = get_cv_result(params=params_dummy_obj_metric_none_verbose, feval=constant_metric)
    assert len(res) == 2
    assert 'valid error-mean' in res

    # no custom objective, no feval
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

    # custom objective, no feval
    # no default metric
    train_booster(params=params_dummy_obj_verbose)
    assert len(evals_result) == 0

    # metric in params
    train_booster(params=params_dummy_obj_metric_log_verbose)
    assert len(evals_result['valid_0']) == 1
    assert 'binary_logloss' in evals_result['valid_0']

    # multiple metrics in params
    train_booster(params=params_dummy_obj_metric_multi_verbose)
    assert len(evals_result['valid_0']) == 2
    assert 'binary_logloss' in evals_result['valid_0']
    assert 'binary_error' in evals_result['valid_0']

    # no custom objective, feval
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

    # custom objective, feval
    # no default metric, only custom one
    train_booster(params=params_dummy_obj_verbose, feval=constant_metric)
    assert len(evals_result['valid_0']) == 1
    assert 'error' in evals_result['valid_0']

    # metric in params with custom one
    train_booster(params=params_dummy_obj_metric_log_verbose, feval=constant_metric)
    assert len(evals_result['valid_0']) == 2
    assert 'binary_logloss' in evals_result['valid_0']
    assert 'error' in evals_result['valid_0']

    # multiple metrics in params with custom one
    train_booster(params=params_dummy_obj_metric_multi_verbose, feval=constant_metric)
    assert len(evals_result['valid_0']) == 3
    assert 'binary_logloss' in evals_result['valid_0']
    assert 'binary_error' in evals_result['valid_0']
    assert 'error' in evals_result['valid_0']

    # custom metric is evaluated despite 'None' is passed
    train_booster(params=params_dummy_obj_metric_none_verbose, feval=constant_metric)
    assert len(evals_result) == 1
    assert 'error' in evals_result['valid_0']

    X, y = load_digits(n_class=3, return_X_y=True)
    lgb_train = lgb.Dataset(X, y)

    obj_multi_aliases = ['multiclass', 'softmax', 'multiclassova', 'multiclass_ova', 'ova', 'ovr']
    for obj_multi_alias in obj_multi_aliases:
        # Custom objective replaces multiclass
        params_obj_class_3_verbose = {'objective': obj_multi_alias, 'num_class': 3, 'verbose': -1}
        params_dummy_obj_class_3_verbose = {'objective': dummy_obj, 'num_class': 3, 'verbose': -1}
        params_dummy_obj_class_1_verbose = {'objective': dummy_obj, 'num_class': 1, 'verbose': -1}
        params_obj_verbose = {'objective': obj_multi_alias, 'verbose': -1}
        params_dummy_obj_verbose = {'objective': dummy_obj, 'verbose': -1}
        # multiclass default metric
        res = get_cv_result(params_obj_class_3_verbose)
        assert len(res) == 2
        assert 'valid multi_logloss-mean' in res
        # multiclass default metric with custom one
        res = get_cv_result(params_obj_class_3_verbose, feval=constant_metric)
        assert len(res) == 4
        assert 'valid multi_logloss-mean' in res
        assert 'valid error-mean' in res
        # multiclass metric alias with custom one for custom objective
        res = get_cv_result(params_dummy_obj_class_3_verbose, feval=constant_metric)
        assert len(res) == 2
        assert 'valid error-mean' in res
        # no metric for invalid class_num
        res = get_cv_result(params_dummy_obj_class_1_verbose)
        assert len(res) == 0
        # custom metric for invalid class_num
        res = get_cv_result(params_dummy_obj_class_1_verbose, feval=constant_metric)
        assert len(res) == 2
        assert 'valid error-mean' in res
        # multiclass metric alias with custom one with invalid class_num
        with pytest.raises(lgb.basic.LightGBMError):
            get_cv_result(params_dummy_obj_class_1_verbose, metrics=obj_multi_alias,
                          feval=constant_metric)
        # multiclass default metric without num_class
        with pytest.raises(lgb.basic.LightGBMError):
            get_cv_result(params_obj_verbose)
        for metric_multi_alias in obj_multi_aliases + ['multi_logloss']:
            # multiclass metric alias
            res = get_cv_result(params_obj_class_3_verbose, metrics=metric_multi_alias)
            assert len(res) == 2
            assert 'valid multi_logloss-mean' in res
        # multiclass metric
        res = get_cv_result(params_obj_class_3_verbose, metrics='multi_error')
        assert len(res) == 2
        assert 'valid multi_error-mean' in res
        # non-valid metric for multiclass objective
        with pytest.raises(lgb.basic.LightGBMError):
            get_cv_result(params_obj_class_3_verbose, metrics='binary_logloss')
    params_class_3_verbose = {'num_class': 3, 'verbose': -1}
    # non-default num_class for default objective
    with pytest.raises(lgb.basic.LightGBMError):
        get_cv_result(params_class_3_verbose)
    # no metric with non-default num_class for custom objective
    res = get_cv_result(params_dummy_obj_class_3_verbose)
    assert len(res) == 0
    for metric_multi_alias in obj_multi_aliases + ['multi_logloss']:
        # multiclass metric alias for custom objective
        res = get_cv_result(params_dummy_obj_class_3_verbose, metrics=metric_multi_alias)
        assert len(res) == 2
        assert 'valid multi_logloss-mean' in res
    # multiclass metric for custom objective
    res = get_cv_result(params_dummy_obj_class_3_verbose, metrics='multi_error')
    assert len(res) == 2
    assert 'valid multi_error-mean' in res
    # binary metric with non-default num_class for custom objective
    with pytest.raises(lgb.basic.LightGBMError):
        get_cv_result(params_dummy_obj_class_3_verbose, metrics='binary_error')


def test_multiple_feval_train():
    X, y = load_breast_cancer(return_X_y=True)

    params = {'verbose': -1, 'objective': 'binary', 'metric': 'binary_logloss'}

    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2)

    train_dataset = lgb.Dataset(data=X_train, label=y_train)
    validation_dataset = lgb.Dataset(data=X_validation, label=y_validation, reference=train_dataset)
    evals_result = {}
    lgb.train(
        params=params,
        train_set=train_dataset,
        valid_sets=validation_dataset,
        num_boost_round=5,
        feval=[constant_metric, decreasing_metric],
        callbacks=[lgb.record_evaluation(evals_result)]
    )

    assert len(evals_result['valid_0']) == 3
    assert 'binary_logloss' in evals_result['valid_0']
    assert 'error' in evals_result['valid_0']
    assert 'decreasing_metric' in evals_result['valid_0']


def test_objective_callable_train_binary_classification():
    X, y = load_breast_cancer(return_X_y=True)
    params = {
        'verbose': -1,
        'objective': logloss_obj,
        'learning_rate': 0.01
    }
    train_dataset = lgb.Dataset(X, y)
    booster = lgb.train(
        params=params,
        train_set=train_dataset,
        num_boost_round=20
    )
    y_pred = logistic_sigmoid(booster.predict(X))
    logloss_error = log_loss(y, y_pred)
    rocauc_error = roc_auc_score(y, y_pred)
    assert booster.params['objective'] == 'none'
    assert logloss_error == pytest.approx(0.547907)
    assert rocauc_error == pytest.approx(0.995944)


def test_objective_callable_train_regression():
    X, y = make_synthetic_regression()
    params = {
        'verbose': -1,
        'objective': mse_obj
    }
    lgb_train = lgb.Dataset(X, y)
    booster = lgb.train(
        params,
        lgb_train,
        num_boost_round=20
    )
    y_pred = booster.predict(X)
    mse_error = mean_squared_error(y, y_pred)
    assert booster.params['objective'] == 'none'
    assert mse_error == pytest.approx(286.724194)


def test_objective_callable_cv_binary_classification():
    X, y = load_breast_cancer(return_X_y=True)
    params = {
        'verbose': -1,
        'objective': logloss_obj,
        'learning_rate': 0.01
    }
    train_dataset = lgb.Dataset(X, y)
    cv_res = lgb.cv(
        params,
        train_dataset,
        num_boost_round=20,
        nfold=3,
        return_cvbooster=True
    )
    cv_booster = cv_res['cvbooster'].boosters
    cv_logloss_errors = [
        log_loss(y, logistic_sigmoid(cb.predict(X))) < 0.56 for cb in cv_booster
    ]
    cv_objs = [
        cb.params['objective'] == 'none' for cb in cv_booster
    ]
    assert all(cv_objs)
    assert all(cv_logloss_errors)


def test_objective_callable_cv_regression():
    X, y = make_synthetic_regression()
    lgb_train = lgb.Dataset(X, y)
    params = {
        'verbose': -1,
        'objective': mse_obj
    }
    cv_res = lgb.cv(
        params,
        lgb_train,
        num_boost_round=20,
        nfold=3,
        stratified=False,
        return_cvbooster=True
    )
    cv_booster = cv_res['cvbooster'].boosters
    cv_mse_errors = [
        mean_squared_error(y, cb.predict(X)) < 463 for cb in cv_booster
    ]
    cv_objs = [
        cb.params['objective'] == 'none' for cb in cv_booster
    ]
    assert all(cv_objs)
    assert all(cv_mse_errors)


def test_multiple_feval_cv():
    X, y = load_breast_cancer(return_X_y=True)

    params = {'verbose': -1, 'objective': 'binary', 'metric': 'binary_logloss'}

    train_dataset = lgb.Dataset(data=X, label=y)

    cv_results = lgb.cv(
        params=params,
        train_set=train_dataset,
        num_boost_round=5,
        feval=[constant_metric, decreasing_metric])

    # Expect three metrics but mean and stdv for each metric
    assert len(cv_results) == 6
    assert 'valid binary_logloss-mean' in cv_results
    assert 'valid error-mean' in cv_results
    assert 'valid decreasing_metric-mean' in cv_results
    assert 'valid binary_logloss-stdv' in cv_results
    assert 'valid error-stdv' in cv_results
    assert 'valid decreasing_metric-stdv' in cv_results


def test_default_objective_and_metric():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    train_dataset = lgb.Dataset(data=X_train, label=y_train)
    validation_dataset = lgb.Dataset(data=X_test, label=y_test, reference=train_dataset)
    evals_result = {}
    params = {'verbose': -1}
    lgb.train(
        params=params,
        train_set=train_dataset,
        valid_sets=validation_dataset,
        num_boost_round=5,
        callbacks=[lgb.record_evaluation(evals_result)]
    )

    assert 'valid_0' in evals_result
    assert len(evals_result['valid_0']) == 1
    assert 'l2' in evals_result['valid_0']
    assert len(evals_result['valid_0']['l2']) == 5


@pytest.mark.parametrize('use_weight', [True, False])
def test_multiclass_custom_objective(use_weight):
    def custom_obj(y_pred, ds):
        y_true = ds.get_label()
        weight = ds.get_weight()
        grad, hess = sklearn_multiclass_custom_objective(y_true, y_pred, weight)
        return grad, hess

    centers = [[-4, -4], [4, 4], [-4, 4]]
    X, y = make_blobs(n_samples=1_000, centers=centers, random_state=42)
    weight = np.full_like(y, 2)
    ds = lgb.Dataset(X, y)
    if use_weight:
        ds.set_weight(weight)
    params = {'objective': 'multiclass', 'num_class': 3, 'num_leaves': 7}
    builtin_obj_bst = lgb.train(params, ds, num_boost_round=10)
    builtin_obj_preds = builtin_obj_bst.predict(X)

    params['objective'] = custom_obj
    custom_obj_bst = lgb.train(params, ds, num_boost_round=10)
    custom_obj_preds = softmax(custom_obj_bst.predict(X))

    np.testing.assert_allclose(builtin_obj_preds, custom_obj_preds, rtol=0.01)


@pytest.mark.parametrize('use_weight', [True, False])
def test_multiclass_custom_eval(use_weight):
    def custom_eval(y_pred, ds):
        y_true = ds.get_label()
        weight = ds.get_weight()  # weight is None when not set
        loss = log_loss(y_true, y_pred, sample_weight=weight)
        return 'custom_logloss', loss, False

    centers = [[-4, -4], [4, 4], [-4, 4]]
    X, y = make_blobs(n_samples=1_000, centers=centers, random_state=42)
    weight = np.full_like(y, 2)
    X_train, X_valid, y_train, y_valid, weight_train, weight_valid = train_test_split(
        X, y, weight, test_size=0.2, random_state=0
    )
    train_ds = lgb.Dataset(X_train, y_train)
    valid_ds = lgb.Dataset(X_valid, y_valid, reference=train_ds)
    if use_weight:
        train_ds.set_weight(weight_train)
        valid_ds.set_weight(weight_valid)
    params = {'objective': 'multiclass', 'num_class': 3, 'num_leaves': 7}
    eval_result = {}
    bst = lgb.train(
        params,
        train_ds,
        num_boost_round=10,
        valid_sets=[train_ds, valid_ds],
        valid_names=['train', 'valid'],
        feval=custom_eval,
        callbacks=[lgb.record_evaluation(eval_result)],
        keep_training_booster=True,
    )

    for key, ds in zip(['train', 'valid'], [train_ds, valid_ds]):
        np.testing.assert_allclose(eval_result[key]['multi_logloss'], eval_result[key]['custom_logloss'])
        _, metric, value, _ = bst.eval(ds, key, feval=custom_eval)[1]  # first element is multi_logloss
        assert metric == 'custom_logloss'
        np.testing.assert_allclose(value, eval_result[key][metric][-1])


@pytest.mark.skipif(psutil.virtual_memory().available / 1024 / 1024 / 1024 < 3, reason='not enough RAM')
def test_model_size():
    X, y = make_synthetic_regression()
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
        before_tree_sizes = model_str[:model_str.find('tree_sizes')]
        trees = model_str[model_str.find('Tree=0'):model_str.find('end of trees')]
        more_trees = (one_tree * multiplier).format(*range(2, total_trees))
        after_trees = model_str[model_str.find('end of trees'):]
        num_end_spaces = 2**31 - one_tree_size * total_trees
        new_model_str = f"{before_tree_sizes}\n\n{trees}{more_trees}{after_trees}{'':{num_end_spaces}}"
        assert len(new_model_str) > 2**31
        bst.model_from_string(new_model_str)
        assert bst.num_trees() == total_trees
        y_pred_new = bst.predict(X, num_iteration=2)
        np.testing.assert_allclose(y_pred, y_pred_new)
    except MemoryError:
        pytest.skipTest('not enough RAM')


@pytest.mark.skipif(getenv('TASK', '') == 'cuda', reason='Skip due to differences in implementation details of CUDA version')
def test_get_split_value_histogram():
    X, y = make_synthetic_regression()
    X = np.repeat(X, 3, axis=0)
    y = np.repeat(y, 3, axis=0)
    X[:, 2] = np.random.default_rng(0).integers(0, 20, size=X.shape[0])
    lgb_train = lgb.Dataset(X, y, categorical_feature=[2])
    gbm = lgb.train({'verbose': -1}, lgb_train, num_boost_round=20)
    # test XGBoost-style return value
    params = {'feature': 0, 'xgboost_style': True}
    assert gbm.get_split_value_histogram(**params).shape == (12, 2)
    assert gbm.get_split_value_histogram(bins=999, **params).shape == (12, 2)
    assert gbm.get_split_value_histogram(bins=-1, **params).shape == (1, 2)
    assert gbm.get_split_value_histogram(bins=0, **params).shape == (1, 2)
    assert gbm.get_split_value_histogram(bins=1, **params).shape == (1, 2)
    assert gbm.get_split_value_histogram(bins=2, **params).shape == (2, 2)
    assert gbm.get_split_value_histogram(bins=6, **params).shape == (6, 2)
    assert gbm.get_split_value_histogram(bins=7, **params).shape == (7, 2)
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
    assert len(hist) == 20
    assert len(bins) == 21
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


@pytest.mark.skipif(getenv('TASK', '') == 'cuda', reason='Skip due to differences in implementation details of CUDA version')
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
        gbm = lgb.train(
            params,
            lgb_train,
            num_boost_round=25,
            valid_sets=valid_sets,
            feval=feval,
            callbacks=[lgb.early_stopping(stopping_rounds=5, first_metric_only=first_metric_only)]
        )
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
        ret = lgb.cv(
            params,
            train_set=lgb_train,
            num_boost_round=25,
            stratified=False,
            feval=feval,
            callbacks=[lgb.early_stopping(stopping_rounds=5, first_metric_only=first_metric_only)],
            eval_train_metric=eval_train_metric
        )
        assert assumed_iteration == len(ret[list(ret.keys())[0]])

    X, y = make_synthetic_regression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test1, X_test2, y_test1, y_test2 = train_test_split(X_test, y_test, test_size=0.5, random_state=73)
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_valid1 = lgb.Dataset(X_test1, y_test1, reference=lgb_train)
    lgb_valid2 = lgb.Dataset(X_test2, y_test2, reference=lgb_train)

    iter_valid1_l1 = 3
    iter_valid1_l2 = 3
    iter_valid2_l1 = 3
    iter_valid2_l2 = 15
    assert len({iter_valid1_l1, iter_valid1_l2, iter_valid2_l1, iter_valid2_l2}) == 2
    iter_min_l1 = min([iter_valid1_l1, iter_valid2_l1])
    iter_min_l2 = min([iter_valid1_l2, iter_valid2_l2])
    iter_min_valid1 = min([iter_valid1_l1, iter_valid1_l2])

    iter_cv_l1 = 15
    iter_cv_l2 = 13
    assert len({iter_cv_l1, iter_cv_l2}) == 2
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
    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=25,
        valid_sets=lgb_eval,
        callbacks=[lgb.record_evaluation(evals_result)]
    )
    ret = log_loss(y_test, gbm.predict(X_test))
    assert ret < 0.14
    assert evals_result['valid_0']['binary_logloss'][-1] == pytest.approx(ret)
    params['feature_fraction'] = 0.5
    gbm2 = lgb.train(params, lgb_train, num_boost_round=25)
    ret2 = log_loss(y_test, gbm2.predict(X_test))
    assert ret != ret2


def test_forced_split_feature_indices(tmp_path):
    X, y = make_synthetic_regression()
    forced_split = {
        "feature": 0,
        "threshold": 0.5,
        "left": {"feature": X.shape[1], "threshold": 0.5},
    }
    tmp_split_file = tmp_path / "forced_split.json"
    with open(tmp_split_file, "w") as f:
        f.write(json.dumps(forced_split))
    lgb_train = lgb.Dataset(X, y)
    params = {
        "objective": "regression",
        "forcedsplits_filename": tmp_split_file
    }
    with pytest.raises(lgb.basic.LightGBMError, match="Forced splits file includes feature index"):
        lgb.train(params, lgb_train)


def test_forced_bins():
    x = np.empty((100, 2))
    x[:, 0] = np.arange(0, 1, 0.01)
    x[:, 1] = -np.arange(0, 1, 0.01)
    y = np.arange(0, 1, 0.01)
    forcedbins_filename = (
        Path(__file__).absolute().parents[2] / 'examples' / 'regression' / 'forced_bins.json'
    )
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
    params['forcedbins_filename'] = (
        Path(__file__).absolute().parents[2] / 'examples' / 'regression' / 'forced_bins2.json'
    )
    params['max_bin'] = 11
    lgb_x = lgb.Dataset(x[:, :1], label=y)
    est = lgb.train(params, lgb_x, num_boost_round=50)
    predicted = est.predict(x[1:, :1])
    _, counts = np.unique(predicted, return_counts=True)
    assert min(counts) >= 9
    assert max(counts) <= 11


def test_binning_same_sign():
    # test that binning works properly for features with only positive or only negative values
    x = np.empty((99, 2))
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
                      "precise_float_parser": True,
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
                           "linear_tree": True,
                           "precise_float_parser": False}
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
        if key != "forcedbins_filename":
            param_name = key
        else:
            param_name = "forced bins"
        err_msg = ("Reducing `min_data_in_leaf` with `feature_pre_filter=true` may cause *"
                   if key == "min_data_in_leaf"
                   else f"Cannot change {param_name} *")
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
    X, y = make_synthetic_regression()
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
    X, y = make_synthetic_regression()
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
        cols = [f'Column_{i}' for i in range(X.shape[1])]
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
    X, y = make_synthetic_regression(n_samples=200)
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
    est = lgb.train(dict(params, interaction_constraints=[[0, 2], [1, 3]]), train_data, num_boost_round=10)
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


def test_linear_trees_num_threads():
    # check that number of threads does not affect result
    np.random.seed(0)
    x = np.arange(0, 1000, 0.1)
    y = 2 * x + np.random.normal(0, 0.1, len(x))
    x = x[:, np.newaxis]
    lgb_train = lgb.Dataset(x, label=y)
    params = {'verbose': -1,
              'objective': 'regression',
              'seed': 0,
              'linear_tree': True,
              'num_threads': 2}
    est = lgb.train(params, lgb_train, num_boost_round=100)
    pred1 = est.predict(x)
    params["num_threads"] = 4
    est = lgb.train(params, lgb_train, num_boost_round=100)
    pred2 = est.predict(x)
    np.testing.assert_allclose(pred1, pred2)


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
    est = lgb.train(
        dict(
            params,
            linear_tree=True
        ),
        lgb_train,
        num_boost_round=10,
        valid_sets=[lgb_train],
        valid_names=['train'],
        callbacks=[lgb.record_evaluation(res)]
    )
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
    est = lgb.train(
        dict(
            params,
            linear_tree=True
        ),
        lgb_train,
        num_boost_round=10,
        valid_sets=[lgb_train],
        valid_names=['train'],
        callbacks=[lgb.record_evaluation(res)]
    )
    pred2 = est.predict(x)
    assert res['train']['l2'][-1] == pytest.approx(mean_squared_error(y, pred2), abs=1e-1)
    assert mean_squared_error(y, pred2) < mean_squared_error(y, pred1)
    # test again with bagging
    res = {}
    est = lgb.train(
        dict(
            params,
            linear_tree=True,
            subsample=0.8,
            bagging_freq=1
        ),
        lgb_train,
        num_boost_round=10,
        valid_sets=[lgb_train],
        valid_names=['train'],
        callbacks=[lgb.record_evaluation(res)]
    )
    pred = est.predict(x)
    assert res['train']['l2'][-1] == pytest.approx(mean_squared_error(y, pred), abs=1e-1)
    # test with a feature that has only one non-nan value
    x = np.concatenate([np.ones([x.shape[0], 1]), x], 1)
    x[500:, 1] = np.nan
    y[500:] += 10
    lgb_train = lgb.Dataset(x, label=y)
    res = {}
    est = lgb.train(
        dict(
            params,
            linear_tree=True,
            subsample=0.8,
            bagging_freq=1
        ),
        lgb_train,
        num_boost_round=10,
        valid_sets=[lgb_train],
        valid_names=['train'],
        callbacks=[lgb.record_evaluation(res)]
    )
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
        callbacks = [lgb.early_stopping(early_stopping_rounds)] if early_stopping_rounds is not None else []
        booster = lgb.train(
            params,
            train_data,
            num_boost_round=50,
            valid_sets=[valid_data],
            callbacks=callbacks
        )

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
    X, y = make_synthetic_regression()
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
    est = lgb.train(
        params,
        lgb_X,
        num_boost_round=10,
        valid_sets=[lgb_X],
        callbacks=[lgb.record_evaluation(res)]
    )
    ap = res['training']['average_precision'][-1]
    pred = est.predict(X)
    sklearn_ap = average_precision_score(y, pred)
    assert ap == pytest.approx(sklearn_ap)
    # test that average precision is 1 where model predicts perfectly
    y = y.copy()
    y[:] = 1
    lgb_X = lgb.Dataset(X, label=y)
    lgb.train(
        params,
        lgb_X,
        num_boost_round=1,
        valid_sets=[lgb_X],
        callbacks=[lgb.record_evaluation(res)]
    )
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


def test_dump_model_hook():

    def hook(obj):
        if 'leaf_value' in obj:
            obj['LV'] = obj['leaf_value']
            del obj['leaf_value']
        return obj

    X, y = load_breast_cancer(return_X_y=True)
    train_data = lgb.Dataset(X, label=y)
    params = {
        "objective": "binary",
        "verbose": -1
    }
    bst = lgb.train(params, train_data, num_boost_round=5)
    dumped_model_str = str(bst.dump_model(5, 0, object_hook=hook))
    assert "leaf_value" not in dumped_model_str
    assert "LV" in dumped_model_str


@pytest.mark.skipif(getenv('TASK', '') == 'cuda', reason='Forced splits are not yet supported by CUDA version')
def test_force_split_with_feature_fraction(tmp_path):
    X, y = make_synthetic_regression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    lgb_train = lgb.Dataset(X_train, y_train)

    forced_split = {
        "feature": 0,
        "threshold": 0.5,
        "right": {
            "feature": 2,
            "threshold": 10.0
        }
    }

    tmp_split_file = tmp_path / "forced_split.json"
    with open(tmp_split_file, "w") as f:
        f.write(json.dumps(forced_split))

    params = {
        "objective": "regression",
        "feature_fraction": 0.6,
        "force_col_wise": True,
        "feature_fraction_seed": 1,
        "forcedsplits_filename": tmp_split_file
    }

    gbm = lgb.train(params, lgb_train)
    ret = mean_absolute_error(y_test, gbm.predict(X_test))
    assert ret < 15.7

    tree_info = gbm.dump_model()["tree_info"]
    assert len(tree_info) > 1
    for tree in tree_info:
        tree_structure = tree["tree_structure"]
        assert tree_structure['split_feature'] == 0


def test_goss_boosting_and_strategy_equivalent():
    X, y = make_synthetic_regression(n_samples=10_000, n_features=10, n_informative=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    base_params = {
        'metric': 'l2',
        'verbose': -1,
        'bagging_seed': 0,
        'learning_rate': 0.05,
        'num_threads': 1,
        'force_row_wise': True,
        'gpu_use_dp': True,
    }
    params1 = {**base_params, 'boosting': 'goss'}
    evals_result1 = {}
    lgb.train(params1, lgb_train,
              num_boost_round=10,
              valid_sets=lgb_eval,
              callbacks=[lgb.record_evaluation(evals_result1)])
    params2 = {**base_params, 'data_sample_strategy': 'goss'}
    evals_result2 = {}
    lgb.train(params2, lgb_train,
              num_boost_round=10,
              valid_sets=lgb_eval,
              callbacks=[lgb.record_evaluation(evals_result2)])
    assert evals_result1['valid_0']['l2'] == evals_result2['valid_0']['l2']


def test_sample_strategy_with_boosting():
    X, y = make_synthetic_regression(n_samples=10_000, n_features=10, n_informative=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    base_params = {
        'metric': 'l2',
        'verbose': -1,
        'num_threads': 1,
        'force_row_wise': True,
        'gpu_use_dp': True,
    }

    params1 = {**base_params, 'boosting': 'dart', 'data_sample_strategy': 'goss'}
    evals_result = {}
    gbm = lgb.train(params1, lgb_train,
                    num_boost_round=10,
                    valid_sets=lgb_eval,
                    callbacks=[lgb.record_evaluation(evals_result)])
    eval_res1 = evals_result['valid_0']['l2'][-1]
    test_res1 = mean_squared_error(y_test, gbm.predict(X_test))
    assert test_res1 == pytest.approx(3149.393862, abs=1.0)
    assert eval_res1 == pytest.approx(test_res1)

    params2 = {**base_params, 'boosting': 'gbdt', 'data_sample_strategy': 'goss'}
    evals_result = {}
    gbm = lgb.train(params2, lgb_train,
                    num_boost_round=10,
                    valid_sets=lgb_eval,
                    callbacks=[lgb.record_evaluation(evals_result)])
    eval_res2 = evals_result['valid_0']['l2'][-1]
    test_res2 = mean_squared_error(y_test, gbm.predict(X_test))
    assert test_res2 == pytest.approx(2547.715968, abs=1.0)
    assert eval_res2 == pytest.approx(test_res2)

    params3 = {**base_params, 'boosting': 'goss', 'data_sample_strategy': 'goss'}
    evals_result = {}
    gbm = lgb.train(params3, lgb_train,
                    num_boost_round=10,
                    valid_sets=lgb_eval,
                    callbacks=[lgb.record_evaluation(evals_result)])
    eval_res3 = evals_result['valid_0']['l2'][-1]
    test_res3 = mean_squared_error(y_test, gbm.predict(X_test))
    assert test_res3 == pytest.approx(2547.715968, abs=1.0)
    assert eval_res3 == pytest.approx(test_res3)

    params4 = {**base_params, 'boosting': 'rf', 'data_sample_strategy': 'goss'}
    evals_result = {}
    gbm = lgb.train(params4, lgb_train,
                    num_boost_round=10,
                    valid_sets=lgb_eval,
                    callbacks=[lgb.record_evaluation(evals_result)])
    eval_res4 = evals_result['valid_0']['l2'][-1]
    test_res4 = mean_squared_error(y_test, gbm.predict(X_test))
    assert test_res4 == pytest.approx(2095.538735, abs=1.0)
    assert eval_res4 == pytest.approx(test_res4)

    assert test_res1 != test_res2
    assert eval_res1 != eval_res2
    assert test_res2 == test_res3
    assert eval_res2 == eval_res3
    assert eval_res1 != eval_res4
    assert test_res1 != test_res4
    assert eval_res2 != eval_res4
    assert test_res2 != test_res4

    params5 = {**base_params, 'boosting': 'dart', 'data_sample_strategy': 'bagging', 'bagging_freq': 1, 'bagging_fraction': 0.5}
    evals_result = {}
    gbm = lgb.train(params5, lgb_train,
                    num_boost_round=10,
                    valid_sets=lgb_eval,
                    callbacks=[lgb.record_evaluation(evals_result)])
    eval_res5 = evals_result['valid_0']['l2'][-1]
    test_res5 = mean_squared_error(y_test, gbm.predict(X_test))
    assert test_res5 == pytest.approx(3134.866931, abs=1.0)
    assert eval_res5 == pytest.approx(test_res5)

    params6 = {**base_params, 'boosting': 'gbdt', 'data_sample_strategy': 'bagging', 'bagging_freq': 1, 'bagging_fraction': 0.5}
    evals_result = {}
    gbm = lgb.train(params6, lgb_train,
                    num_boost_round=10,
                    valid_sets=lgb_eval,
                    callbacks=[lgb.record_evaluation(evals_result)])
    eval_res6 = evals_result['valid_0']['l2'][-1]
    test_res6 = mean_squared_error(y_test, gbm.predict(X_test))
    assert test_res6 == pytest.approx(2539.792378, abs=1.0)
    assert eval_res6 == pytest.approx(test_res6)
    assert test_res5 != test_res6
    assert eval_res5 != eval_res6

    params7 = {**base_params, 'boosting': 'rf', 'data_sample_strategy': 'bagging', 'bagging_freq': 1, 'bagging_fraction': 0.5}
    evals_result = {}
    gbm = lgb.train(params7, lgb_train,
                    num_boost_round=10,
                    valid_sets=lgb_eval,
                    callbacks=[lgb.record_evaluation(evals_result)])
    eval_res7 = evals_result['valid_0']['l2'][-1]
    test_res7 = mean_squared_error(y_test, gbm.predict(X_test))
    assert test_res7 == pytest.approx(1518.704481, abs=1.0)
    assert eval_res7 == pytest.approx(test_res7)
    assert test_res5 != test_res7
    assert eval_res5 != eval_res7
    assert test_res6 != test_res7
    assert eval_res6 != eval_res7


def test_record_evaluation_with_train():
    X, y = make_synthetic_regression()
    ds = lgb.Dataset(X, y)
    eval_result = {}
    callbacks = [lgb.record_evaluation(eval_result)]
    params = {'objective': 'l2', 'num_leaves': 3}
    num_boost_round = 5
    bst = lgb.train(params, ds, num_boost_round=num_boost_round, valid_sets=[ds], callbacks=callbacks)
    assert list(eval_result.keys()) == ['training']
    train_mses = []
    for i in range(num_boost_round):
        pred = bst.predict(X, num_iteration=i + 1)
        mse = mean_squared_error(y, pred)
        train_mses.append(mse)
    np.testing.assert_allclose(eval_result['training']['l2'], train_mses)


@pytest.mark.parametrize('train_metric', [False, True])
def test_record_evaluation_with_cv(train_metric):
    X, y = make_synthetic_regression()
    ds = lgb.Dataset(X, y)
    eval_result = {}
    callbacks = [lgb.record_evaluation(eval_result)]
    metrics = ['l2', 'rmse']
    params = {'objective': 'l2', 'num_leaves': 3, 'metric': metrics}
    cv_hist = lgb.cv(params, ds, num_boost_round=5, stratified=False, callbacks=callbacks, eval_train_metric=train_metric)
    expected_datasets = {'valid'}
    if train_metric:
        expected_datasets.add('train')
    assert set(eval_result.keys()) == expected_datasets
    for dataset in expected_datasets:
        for metric in metrics:
            for agg in ('mean', 'stdv'):
                key = f'{dataset} {metric}-{agg}'
                np.testing.assert_allclose(
                    cv_hist[key], eval_result[dataset][f'{metric}-{agg}']
                )


def test_pandas_with_numpy_regular_dtypes():
    pd = pytest.importorskip('pandas')
    uints = ['uint8', 'uint16', 'uint32', 'uint64']
    ints = ['int8', 'int16', 'int32', 'int64']
    bool_and_floats = ['bool', 'float16', 'float32', 'float64']
    rng = np.random.RandomState(42)

    n_samples = 100
    # data as float64
    df = pd.DataFrame({
        'x1': rng.randint(0, 2, n_samples),
        'x2': rng.randint(1, 3, n_samples),
        'x3': 10 * rng.randint(1, 3, n_samples),
        'x4': 100 * rng.randint(1, 3, n_samples),
    })
    df = df.astype(np.float64)
    y = df['x1'] * (df['x2'] + df['x3'] + df['x4'])
    ds = lgb.Dataset(df, y)
    params = {'objective': 'l2', 'num_leaves': 31, 'min_child_samples': 1}
    bst = lgb.train(params, ds, num_boost_round=5)
    preds = bst.predict(df)

    # test all features were used
    assert bst.trees_to_dataframe()['split_feature'].nunique() == df.shape[1]
    # test the score is better than predicting the mean
    baseline = np.full_like(y, y.mean())
    assert mean_squared_error(y, preds) < mean_squared_error(y, baseline)

    # test all predictions are equal using different input dtypes
    for target_dtypes in [uints, ints, bool_and_floats]:
        df2 = df.astype({f'x{i}': dtype for i, dtype in enumerate(target_dtypes, start=1)})
        assert df2.dtypes.tolist() == target_dtypes
        ds2 = lgb.Dataset(df2, y)
        bst2 = lgb.train(params, ds2, num_boost_round=5)
        preds2 = bst2.predict(df2)
        np.testing.assert_allclose(preds, preds2)


def test_pandas_nullable_dtypes():
    pd = pytest.importorskip('pandas')
    rng = np.random.RandomState(0)
    df = pd.DataFrame({
        'x1': rng.randint(1, 3, size=100),
        'x2': np.linspace(-1, 1, 100),
        'x3': pd.arrays.SparseArray(rng.randint(0, 11, size=100)),
        'x4': rng.rand(100) < 0.5,
    })
    # introduce some missing values
    df.loc[1, 'x1'] = np.nan
    df.loc[2, 'x2'] = np.nan
    df.loc[3, 'x4'] = np.nan
    # the previous line turns x3 into object dtype in recent versions of pandas
    df['x4'] = df['x4'].astype(np.float64)
    y = df['x1'] * df['x2'] + df['x3'] * (1 + df['x4'])
    y = y.fillna(0)

    # train with regular dtypes
    params = {'objective': 'l2', 'num_leaves': 31, 'min_child_samples': 1}
    ds = lgb.Dataset(df, y)
    bst = lgb.train(params, ds, num_boost_round=5)
    preds = bst.predict(df)

    # convert to nullable dtypes
    df2 = df.copy()
    df2['x1'] = df2['x1'].astype('Int32')
    df2['x2'] = df2['x2'].astype('Float64')
    df2['x4'] = df2['x4'].astype('boolean')

    # test training succeeds
    ds_nullable_dtypes = lgb.Dataset(df2, y)
    bst_nullable_dtypes = lgb.train(params, ds_nullable_dtypes, num_boost_round=5)
    preds_nullable_dtypes = bst_nullable_dtypes.predict(df2)

    trees_df = bst_nullable_dtypes.trees_to_dataframe()
    # test all features were used
    assert trees_df['split_feature'].nunique() == df.shape[1]
    # test the score is better than predicting the mean
    baseline = np.full_like(y, y.mean())
    assert mean_squared_error(y, preds) < mean_squared_error(y, baseline)

    # test equal predictions
    np.testing.assert_allclose(preds, preds_nullable_dtypes)


def test_boost_from_average_with_single_leaf_trees():
    # test data are taken from bug report
    # https://github.com/microsoft/LightGBM/issues/4708
    X = np.array([
        [1021.0589, 1018.9578],
        [1023.85754, 1018.7854],
        [1024.5468, 1018.88513],
        [1019.02954, 1018.88513],
        [1016.79926, 1018.88513],
        [1007.6, 1018.88513]], dtype=np.float32)
    y = np.array([1023.8, 1024.6, 1024.4, 1023.8, 1022.0, 1014.4], dtype=np.float32)
    params = {
        "extra_trees": True,
        "min_data_in_bin": 1,
        "extra_seed": 7,
        "objective": "regression",
        "verbose": -1,
        "boost_from_average": True,
        "min_data_in_leaf": 1,
    }
    train_set = lgb.Dataset(X, y)
    model = lgb.train(params=params, train_set=train_set, num_boost_round=10)

    preds = model.predict(X)
    mean_preds = np.mean(preds)
    assert y.min() <= mean_preds <= y.max()


def test_cegb_split_buffer_clean():
    # modified from https://github.com/microsoft/LightGBM/issues/3679#issuecomment-938652811
    # and https://github.com/microsoft/LightGBM/pull/5087
    # test that the ``splits_per_leaf_`` of CEGB is cleaned before training a new tree
    # which is done in the fix #5164
    # without the fix:
    #    Check failed: (best_split_info.left_count) > (0)

    R, C = 1000, 100
    seed = 29
    np.random.seed(seed)
    data = np.random.randn(R, C)
    for i in range(1, C):
        data[i] += data[0] * np.random.randn()

    N = int(0.8 * len(data))
    train_data = data[:N]
    test_data = data[N:]
    train_y = np.sum(train_data, axis=1)
    test_y = np.sum(test_data, axis=1)

    train = lgb.Dataset(train_data, train_y, free_raw_data=True)

    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'max_bin': 255,
        'num_leaves': 31,
        'seed': 0,
        'learning_rate': 0.1,
        'min_data_in_leaf': 0,
        'verbose': -1,
        'min_split_gain': 1000.0,
        'cegb_penalty_feature_coupled': 5 * np.arange(C),
        'cegb_penalty_split': 0.0002,
        'cegb_tradeoff': 10.0,
        'force_col_wise': True,
    }

    model = lgb.train(params, train, num_boost_round=10)
    predicts = model.predict(test_data)
    rmse = np.sqrt(mean_squared_error(test_y, predicts))
    assert rmse < 10.0


def test_verbosity_and_verbose(capsys):
    X, y = make_synthetic_regression()
    ds = lgb.Dataset(X, y)
    params = {
        'num_leaves': 3,
        'verbose': 1,
        'verbosity': 0,
    }
    lgb.train(params, ds, num_boost_round=1)
    expected_msg = (
        '[LightGBM] [Warning] verbosity is set=0, verbose=1 will be ignored. '
        'Current value: verbosity=0'
    )
    stdout = capsys.readouterr().out
    assert expected_msg in stdout


@pytest.mark.parametrize('verbosity_param', lgb.basic._ConfigAliases.get("verbosity"))
@pytest.mark.parametrize('verbosity', [-1, 0])
def test_verbosity_can_suppress_alias_warnings(capsys, verbosity_param, verbosity):
    X, y = make_synthetic_regression()
    ds = lgb.Dataset(X, y)
    params = {
        'num_leaves': 3,
        'subsample': 0.75,
        'bagging_fraction': 0.8,
        'force_col_wise': True,
        verbosity_param: verbosity,
    }
    lgb.train(params, ds, num_boost_round=1)
    expected_msg = (
        '[LightGBM] [Warning] bagging_fraction is set=0.8, subsample=0.75 will be ignored. '
        'Current value: bagging_fraction=0.8'
    )
    stdout = capsys.readouterr().out
    if verbosity >= 0:
        assert expected_msg in stdout
    else:
        assert re.search(r'\[LightGBM\]', stdout) is None


@pytest.mark.skipif(not PANDAS_INSTALLED, reason='pandas is not installed')
def test_validate_features():
    X, y = make_synthetic_regression()
    features = ['x1', 'x2', 'x3', 'x4']
    df = pd_DataFrame(X, columns=features)
    ds = lgb.Dataset(df, y)
    bst = lgb.train({'num_leaves': 15, 'verbose': -1}, ds, num_boost_round=10)
    assert bst.feature_name() == features

    # try to predict with a different feature
    df2 = df.rename(columns={'x3': 'z'})
    with pytest.raises(lgb.basic.LightGBMError, match="Expected 'x3' at position 2 but found 'z'"):
        bst.predict(df2, validate_features=True)

    # check that disabling the check doesn't raise the error
    bst.predict(df2, validate_features=False)

    # try to refit with a different feature
    with pytest.raises(lgb.basic.LightGBMError, match="Expected 'x3' at position 2 but found 'z'"):
        bst.refit(df2, y, validate_features=True)

    # check that disabling the check doesn't raise the error
    bst.refit(df2, y, validate_features=False)


def test_train_and_cv_raise_informative_error_for_train_set_of_wrong_type():
    with pytest.raises(TypeError, match=r"train\(\) only accepts Dataset object, train_set has type 'list'\."):
        lgb.train({}, train_set=[])
    with pytest.raises(TypeError, match=r"cv\(\) only accepts Dataset object, train_set has type 'list'\."):
        lgb.cv({}, train_set=[])


@pytest.mark.parametrize('num_boost_round', [-7, -1, 0])
def test_train_and_cv_raise_informative_error_for_impossible_num_boost_round(num_boost_round):
    X, y = make_synthetic_regression(n_samples=100)
    error_msg = rf"num_boost_round must be greater than 0\. Got {num_boost_round}\."
    with pytest.raises(ValueError, match=error_msg):
        lgb.train({}, train_set=lgb.Dataset(X, y), num_boost_round=num_boost_round)
    with pytest.raises(ValueError, match=error_msg):
        lgb.cv({}, train_set=lgb.Dataset(X, y), num_boost_round=num_boost_round)


def test_train_raises_informative_error_if_any_valid_sets_are_not_dataset_objects():
    X, y = make_synthetic_regression(n_samples=100)
    X_valid = X * 2.0
    with pytest.raises(TypeError, match=r"Every item in valid_sets must be a Dataset object\. Item 1 has type 'tuple'\."):
        lgb.train(
            params={},
            train_set=lgb.Dataset(X, y),
            valid_sets=[
                lgb.Dataset(X_valid, y),
                ([1.0], [2.0]),
                [5.6, 5.7, 5.8]
            ]
        )


def test_train_raises_informative_error_for_params_of_wrong_type():
    X, y = make_synthetic_regression()
    params = {"early_stopping_round": "too-many"}
    dtrain = lgb.Dataset(X, label=y)
    with pytest.raises(lgb.basic.LightGBMError, match="Parameter early_stopping_round should be of type int, got \"too-many\""):
        lgb.train(params, dtrain)


def test_quantized_training():
    X, y = make_synthetic_regression()
    ds = lgb.Dataset(X, label=y)
    bst_params = {'num_leaves': 15, 'verbose': -1, 'seed': 0}
    bst = lgb.train(bst_params, ds, num_boost_round=10)
    rmse = np.sqrt(np.mean((bst.predict(X) - y) ** 2))
    bst_params.update({
        'use_quantized_grad': True,
        'num_grad_quant_bins': 30,
        'quant_train_renew_leaf': True,
    })
    quant_bst = lgb.train(bst_params, ds, num_boost_round=10)
    quant_rmse = np.sqrt(np.mean((quant_bst.predict(X) - y) ** 2))
    assert quant_rmse < rmse + 6.0
