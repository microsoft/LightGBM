# coding: utf-8
"""Tests for lightgbm.dask module"""

import socket
from itertools import groupby
from os import getenv
from sys import platform

import lightgbm as lgb
import pytest
if not platform.startswith('linux'):
    pytest.skip('lightgbm.dask is currently supported in Linux environments', allow_module_level=True)
if not lgb.compat.DASK_INSTALLED:
    pytest.skip('Dask is not installed', allow_module_level=True)

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from dask.array.utils import assert_eq
from distributed.utils_test import client, cluster_fixture, gen_cluster, loop
from scipy.sparse import csr_matrix
from sklearn.datasets import make_blobs, make_regression
from sklearn.utils import check_random_state

from .utils import make_ranking


# time, in seconds, to wait for the Dask client to close. Used to avoid teardown errors
# see https://distributed.dask.org/en/latest/api.html#distributed.Client.close
CLIENT_CLOSE_TIMEOUT = 60

data_output = ['array', 'scipy_csr_matrix', 'dataframe']
data_centers = [[[-4, -4], [4, 4]], [[-4, -4], [4, 4], [-4, 4]]]
group_sizes = [5, 5, 5, 10, 10, 10, 20, 20, 20, 50, 50]

pytestmark = [
    pytest.mark.skipif(getenv('TASK', '') == 'mpi', reason='Fails to run with MPI interface'),
    pytest.mark.skipif(getenv('TASK', '') == 'gpu', reason='Fails to run with GPU interface')
]


@pytest.fixture()
def listen_port():
    listen_port.port += 10
    return listen_port.port


listen_port.port = 13000


def _create_ranking_data(n_samples=100, output='array', chunk_size=50, **kwargs):
    X, y, g = make_ranking(n_samples=n_samples, random_state=42, **kwargs)
    rnd = np.random.RandomState(42)
    w = rnd.rand(X.shape[0]) * 0.01
    g_rle = np.array([len(list(grp)) for _, grp in groupby(g)])

    if output == 'dataframe':
        # add target, weight, and group to DataFrame so that partitions abide by group boundaries.
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        X = X_df.copy()
        X_df = X_df.assign(y=y, g=g, w=w)

        # set_index ensures partitions are based on group id.
        # See https://stackoverflow.com/questions/49532824/dask-dataframe-split-partitions-based-on-a-column-or-function.
        X_df.set_index('g', inplace=True)
        dX = dd.from_pandas(X_df, chunksize=chunk_size)

        # separate target, weight from features.
        dy = dX['y']
        dw = dX['w']
        dX = dX.drop(columns=['y', 'w'])
        dg = dX.index.to_series()

        # encode group identifiers into run-length encoding, the format LightGBMRanker is expecting
        # so that within each partition, sum(g) = n_samples.
        dg = dg.map_partitions(lambda p: p.groupby('g', sort=False).apply(lambda z: z.shape[0]))
    elif output == 'array':
        # ranking arrays: one chunk per group. Each chunk must include all columns.
        p = X.shape[1]
        dX, dy, dw, dg = [], [], [], []
        for g_idx, rhs in enumerate(np.cumsum(g_rle)):
            lhs = rhs - g_rle[g_idx]
            dX.append(da.from_array(X[lhs:rhs, :], chunks=(rhs - lhs, p)))
            dy.append(da.from_array(y[lhs:rhs]))
            dw.append(da.from_array(w[lhs:rhs]))
            dg.append(da.from_array(np.array([g_rle[g_idx]])))

        dX = da.concatenate(dX, axis=0)
        dy = da.concatenate(dy, axis=0)
        dw = da.concatenate(dw, axis=0)
        dg = da.concatenate(dg, axis=0)
    else:
        raise ValueError('Ranking data creation only supported for Dask arrays and dataframes')

    return X, y, w, g_rle, dX, dy, dw, dg


def _create_data(objective, n_samples=100, centers=2, output='array', chunk_size=50):
    if objective == 'classification':
        X, y = make_blobs(n_samples=n_samples, centers=centers, random_state=42)
    elif objective == 'regression':
        X, y = make_regression(n_samples=n_samples, random_state=42)
    else:
        raise ValueError("Unknown objective '%s'" % objective)
    rnd = np.random.RandomState(42)
    weights = rnd.random(X.shape[0]) * 0.01

    if output == 'array':
        dX = da.from_array(X, (chunk_size, X.shape[1]))
        dy = da.from_array(y, chunk_size)
        dw = da.from_array(weights, chunk_size)
    elif output == 'dataframe':
        X_df = pd.DataFrame(X, columns=['feature_%d' % i for i in range(X.shape[1])])
        y_df = pd.Series(y, name='target')
        dX = dd.from_pandas(X_df, chunksize=chunk_size)
        dy = dd.from_pandas(y_df, chunksize=chunk_size)
        dw = dd.from_array(weights, chunksize=chunk_size)
    elif output == 'scipy_csr_matrix':
        dX = da.from_array(X, chunks=(chunk_size, X.shape[1])).map_blocks(csr_matrix)
        dy = da.from_array(y, chunks=chunk_size)
        dw = da.from_array(weights, chunk_size)
    else:
        raise ValueError("Unknown output type '%s'" % output)

    return X, y, weights, dX, dy, dw


def _r2_score(dy_true, dy_pred):
    numerator = ((dy_true - dy_pred) ** 2).sum(axis=0, dtype=np.float64)
    denominator = ((dy_true - dy_pred.mean(axis=0)) ** 2).sum(axis=0, dtype=np.float64)
    return (1 - numerator / denominator).compute()


def _accuracy_score(dy_true, dy_pred):
    return da.average(dy_true == dy_pred).compute()


@pytest.mark.parametrize('output', data_output)
@pytest.mark.parametrize('centers', data_centers)
def test_classifier(output, centers, client, listen_port):
    X, y, w, dX, dy, dw = _create_data(
        objective='classification',
        output=output,
        centers=centers
    )

    params = {
        "n_estimators": 10,
        "num_leaves": 10
    }
    dask_classifier = lgb.DaskLGBMClassifier(
        time_out=5,
        local_listen_port=listen_port,
        **params
    )
    dask_classifier = dask_classifier.fit(dX, dy, sample_weight=dw, client=client)
    p1 = dask_classifier.predict(dX)
    p1_proba = dask_classifier.predict_proba(dX).compute()
    p1_local = dask_classifier.to_local().predict(X)
    s1 = _accuracy_score(dy, p1)
    p1 = p1.compute()

    local_classifier = lgb.LGBMClassifier(**params)
    local_classifier.fit(X, y, sample_weight=w)
    p2 = local_classifier.predict(X)
    p2_proba = local_classifier.predict_proba(X)
    s2 = local_classifier.score(X, y)

    assert_eq(s1, s2)
    assert_eq(p1, p2)
    assert_eq(y, p1)
    assert_eq(y, p2)
    assert_eq(p1_proba, p2_proba, atol=0.3)
    assert_eq(p1_local, p2)
    assert_eq(y, p1_local)

    client.close(timeout=CLIENT_CLOSE_TIMEOUT)


@pytest.mark.parametrize('output', data_output)
@pytest.mark.parametrize('centers', data_centers)
def test_classifier_pred_contrib(output, centers, client, listen_port):
    X, y, w, dX, dy, dw = _create_data(
        objective='classification',
        output=output,
        centers=centers
    )

    params = {
        "n_estimators": 10,
        "num_leaves": 10
    }
    dask_classifier = lgb.DaskLGBMClassifier(
        time_out=5,
        local_listen_port=listen_port,
        tree_learner='data',
        **params
    )
    dask_classifier = dask_classifier.fit(dX, dy, sample_weight=dw, client=client)
    preds_with_contrib = dask_classifier.predict(dX, pred_contrib=True).compute()

    local_classifier = lgb.LGBMClassifier(**params)
    local_classifier.fit(X, y, sample_weight=w)
    local_preds_with_contrib = local_classifier.predict(X, pred_contrib=True)

    if output == 'scipy_csr_matrix':
        preds_with_contrib = np.array(preds_with_contrib.todense())

    # shape depends on whether it is binary or multiclass classification
    num_features = dask_classifier.n_features_
    num_classes = dask_classifier.n_classes_
    if num_classes == 2:
        expected_num_cols = num_features + 1
    else:
        expected_num_cols = (num_features + 1) * num_classes

    # * shape depends on whether it is binary or multiclass classification
    # * matrix for binary classification is of the form [feature_contrib, base_value],
    #   for multi-class it's [feat_contrib_class1, base_value_class1, feat_contrib_class2, base_value_class2, etc.]
    # * contrib outputs for distributed training are different than from local training, so we can just test
    #   that the output has the right shape and base values are in the right position
    assert preds_with_contrib.shape[1] == expected_num_cols
    assert preds_with_contrib.shape == local_preds_with_contrib.shape

    if num_classes == 2:
        assert len(np.unique(preds_with_contrib[:, num_features]) == 1)
    else:
        for i in range(num_classes):
            base_value_col = num_features * (i + 1) + i
            assert len(np.unique(preds_with_contrib[:, base_value_col]) == 1)

    client.close(timeout=CLIENT_CLOSE_TIMEOUT)


def test_training_does_not_fail_on_port_conflicts(client):
    _, _, _, dX, dy, dw = _create_data('classification', output='array')

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 12400))

        dask_classifier = lgb.DaskLGBMClassifier(
            time_out=5,
            local_listen_port=12400,
            n_estimators=5,
            num_leaves=5
        )
        for _ in range(5):
            dask_classifier.fit(
                X=dX,
                y=dy,
                sample_weight=dw,
                client=client
            )
            assert dask_classifier.booster_

    client.close(timeout=CLIENT_CLOSE_TIMEOUT)


@pytest.mark.parametrize('output', data_output)
def test_regressor(output, client, listen_port):
    X, y, w, dX, dy, dw = _create_data(
        objective='regression',
        output=output
    )

    params = {
        "random_state": 42,
        "num_leaves": 10
    }
    dask_regressor = lgb.DaskLGBMRegressor(
        time_out=5,
        local_listen_port=listen_port,
        tree='data',
        **params
    )
    dask_regressor = dask_regressor.fit(dX, dy, client=client, sample_weight=dw)
    p1 = dask_regressor.predict(dX)
    if output != 'dataframe':
        s1 = _r2_score(dy, p1)
    p1 = p1.compute()
    p1_local = dask_regressor.to_local().predict(X)
    s1_local = dask_regressor.to_local().score(X, y)

    local_regressor = lgb.LGBMRegressor(**params)
    local_regressor.fit(X, y, sample_weight=w)
    s2 = local_regressor.score(X, y)
    p2 = local_regressor.predict(X)

    # Scores should be the same
    if output != 'dataframe':
        assert_eq(s1, s2, atol=.01)
        assert_eq(s1, s1_local, atol=.003)

    # Predictions should be roughly the same
    assert_eq(y, p1, rtol=1., atol=100.)
    assert_eq(y, p2, rtol=1., atol=50.)
    assert_eq(p1, p1_local)

    client.close(timeout=CLIENT_CLOSE_TIMEOUT)


@pytest.mark.parametrize('output', data_output)
def test_regressor_pred_contrib(output, client, listen_port):
    X, y, w, dX, dy, dw = _create_data(
        objective='regression',
        output=output
    )

    params = {
        "n_estimators": 10,
        "num_leaves": 10
    }
    dask_regressor = lgb.DaskLGBMRegressor(
        time_out=5,
        local_listen_port=listen_port,
        tree_learner='data',
        **params
    )
    dask_regressor = dask_regressor.fit(dX, dy, sample_weight=dw, client=client)
    preds_with_contrib = dask_regressor.predict(dX, pred_contrib=True).compute()

    local_regressor = lgb.LGBMRegressor(**params)
    local_regressor.fit(X, y, sample_weight=w)
    local_preds_with_contrib = local_regressor.predict(X, pred_contrib=True)

    if output == "scipy_csr_matrix":
        preds_with_contrib = np.array(preds_with_contrib.todense())

    # contrib outputs for distributed training are different than from local training, so we can just test
    # that the output has the right shape and base values are in the right position
    num_features = dX.shape[1]
    assert preds_with_contrib.shape[1] == num_features + 1
    assert preds_with_contrib.shape == local_preds_with_contrib.shape

    client.close(timeout=CLIENT_CLOSE_TIMEOUT)


@pytest.mark.parametrize('output', data_output)
@pytest.mark.parametrize('alpha', [.1, .5, .9])
def test_regressor_quantile(output, client, listen_port, alpha):
    X, y, w, dX, dy, dw = _create_data(
        objective='regression',
        output=output
    )

    params = {
        "objective": "quantile",
        "alpha": alpha,
        "random_state": 42,
        "n_estimators": 10,
        "num_leaves": 10
    }
    dask_regressor = lgb.DaskLGBMRegressor(
        local_listen_port=listen_port,
        tree_learner_type='data_parallel',
        **params
    )
    dask_regressor = dask_regressor.fit(dX, dy, client=client, sample_weight=dw)
    p1 = dask_regressor.predict(dX).compute()
    q1 = np.count_nonzero(y < p1) / y.shape[0]

    local_regressor = lgb.LGBMRegressor(**params)
    local_regressor.fit(X, y, sample_weight=w)
    p2 = local_regressor.predict(X)
    q2 = np.count_nonzero(y < p2) / y.shape[0]

    # Quantiles should be right
    np.testing.assert_allclose(q1, alpha, atol=0.2)
    np.testing.assert_allclose(q2, alpha, atol=0.2)

    client.close(timeout=CLIENT_CLOSE_TIMEOUT)


@pytest.mark.parametrize('output', ['array', 'dataframe'])
@pytest.mark.parametrize('group', [None, group_sizes])
def test_ranker(output, client, listen_port, group):

    X, y, w, g, dX, dy, dw, dg = _create_ranking_data(
        output=output,
        group=group
    )

    # use many trees + leaves to overfit, help ensure that dask data-parallel strategy matches that of
    # serial learner. See https://github.com/microsoft/LightGBM/issues/3292#issuecomment-671288210.
    params = {
        "random_state": 42,
        "n_estimators": 50,
        "num_leaves": 20,
        "min_child_samples": 1
    }
    dask_ranker = lgb.DaskLGBMRanker(
        time_out=5,
        local_listen_port=listen_port,
        tree_learner_type='data_parallel',
        **params
    )
    dask_ranker = dask_ranker.fit(dX, dy, sample_weight=dw, group=dg, client=client)
    rnkvec_dask = dask_ranker.predict(dX)
    rnkvec_dask = rnkvec_dask.compute()
    rnkvec_dask_local = dask_ranker.to_local().predict(X)

    local_ranker = lgb.LGBMRanker(**params)
    local_ranker.fit(X, y, sample_weight=w, group=g)
    rnkvec_local = local_ranker.predict(X)

    # distributed ranker should be able to rank decently well and should
    # have high rank correlation with scores from serial ranker.
    dcor = spearmanr(rnkvec_dask, y).correlation
    assert dcor > 0.6
    assert spearmanr(rnkvec_dask, rnkvec_local).correlation > 0.75
    assert_eq(rnkvec_dask, rnkvec_dask_local)

    client.close(timeout=CLIENT_CLOSE_TIMEOUT)


def test_find_open_port_works():
    worker_ip = '127.0.0.1'
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((worker_ip, 12400))
        new_port = lgb.dask._find_open_port(
            worker_ip=worker_ip,
            local_listen_port=12400,
            ports_to_skip=set()
        )
        assert new_port == 12401

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s_1:
        s_1.bind((worker_ip, 12400))
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s_2:
            s_2.bind((worker_ip, 12401))
            new_port = lgb.dask._find_open_port(
                worker_ip=worker_ip,
                local_listen_port=12400,
                ports_to_skip=set()
            )
            assert new_port == 12402


def test_warns_and_continues_on_unrecognized_tree_learner(client):
    X = da.random.random((1e3, 10))
    y = da.random.random((1e3, 1))
    dask_regressor = lgb.DaskLGBMRegressor(
        time_out=5,
        local_listen_port=1234,
        tree_learner='some-nonsense-value',
        n_estimators=1,
        num_leaves=2
    )
    with pytest.warns(UserWarning, match='Parameter tree_learner set to some-nonsense-value'):
        dask_regressor = dask_regressor.fit(X, y, client=client)

    assert dask_regressor.fitted_

    client.close(timeout=CLIENT_CLOSE_TIMEOUT)


def test_warns_but_makes_no_changes_for_feature_or_voting_tree_learner(client):
    X = da.random.random((1e3, 10))
    y = da.random.random((1e3, 1))
    for tree_learner in ['feature_parallel', 'voting']:
        dask_regressor = lgb.DaskLGBMRegressor(
            time_out=5,
            local_listen_port=1234,
            tree_learner=tree_learner,
            n_estimators=1,
            num_leaves=2
        )
        with pytest.warns(UserWarning, match='Support for tree_learner %s in lightgbm' % tree_learner):
            dask_regressor = dask_regressor.fit(X, y, client=client)

        assert dask_regressor.fitted_
        assert dask_regressor.get_params()['tree_learner'] == tree_learner

    client.close(timeout=CLIENT_CLOSE_TIMEOUT)


@gen_cluster(client=True, timeout=None)
def test_errors(c, s, a, b):
    def f(part):
        raise Exception('foo')

    df = dd.demo.make_timeseries()
    df = df.map_partitions(f, meta=df._meta)
    with pytest.raises(Exception) as info:
        yield lgb.dask._train(
            client=c,
            data=df,
            label=df.x,
            params={},
            model_factory=lgb.LGBMClassifier
        )
        assert 'foo' in str(info.value)
