# coding: utf-8
"""Tests for lightgbm.dask module

An easy way to run these tests is from the (python) docker container.
Also see lightgbm-dask-testing repo: https://github.com/jameslamb/lightgbm-dask-testing
"""
import itertools
import os
import socket
import sys

import pytest
if not sys.platform.startswith('linux'):
    pytest.skip('lightgbm.dask is currently supported in Linux environments', allow_module_level=True)

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import scipy.sparse
from dask.array.utils import assert_eq
from dask_ml.metrics import accuracy_score, r2_score
from distributed.utils_test import client, cluster_fixture, gen_cluster, loop
from sklearn.datasets import make_blobs, make_regression
from sklearn.utils import check_random_state

import lightgbm
import lightgbm.dask as dlgbm

data_output = ['array', 'scipy_csr_matrix', 'dataframe']
data_centers = [[[-4, -4], [4, 4]], [[-4, -4], [4, 4], [-4, 4]]]
group_sizes = [5, 5, 5, 10, 10, 10, 20, 20, 20, 50, 50]

pytestmark = [
    pytest.mark.skipif(os.getenv('TASK', '') == 'mpi', reason='Fails to run with MPI interface')
]


@pytest.fixture()
def listen_port():
    listen_port.port += 10
    return listen_port.port


listen_port.port = 13000


def _make_ranking(n_samples=100, n_features=20, n_informative=5, gmax=2,
                  group=None, random_gs=False, avg_gs=10, random_state=0):
    """Generate a learning-to-rank dataset - feature vectors grouped together with
    integer-valued graded relevance scores. Replace this with a sklearn.datasets function
    if ranking objective becomes supported in sklearn.datasets module.

    Parameters
    ----------
    n_samples: int (default=100)
        Total number of documents (records) in the dataset
    n_features : int (default=20)
        Total number of features in the dataset
    n_informative : int (default=5)
        Number of features that are "informative" for ranking, as they are bias + beta * unif(min=y, max=y+1),
        where bias and beta are standard normal variates. If this is greater than n_features, the dataset will have
        n_features features, all will be informative.
    group : array-like, optional (default=None)
        1-d array or list of group sizes. When `group` is specified, this overrides n_samples, random_gs, and
        avg_gs by simply creating groups with sizes group[0], ..., group[-1].
    gmax : int (default=2)
        Maximum graded relevance value for creating relevance/target vector. If you set this to 2, for example, all
        documents in a group will have relevance scores of either 0, 1, or 2.
    random_gs : bool (default=False)
        True will make group sizes ~ Poisson(avg_gs), False will make group sizes == avg_gs.
    avg_gs : int (default=10)
        Average number of documents (records) in each group

    Returns
    ----------
    X : 2-d np.ndarray of shape = [n_samples (or np.sum(group), n_features]
        Input feature matrix for ranking objective
    y : 1-d np.array of shape = [n_samples (or np.sum(group))]
        integer-graded relevance scores
    group_ids: 1-d np.array of shape = [n_samples (or np.sum(group))]
        vector of group ids, each value indicates to which group each record belongs
    """
    rnd_generator = check_random_state(random_state)

    y_vec, group_id_vec = np.empty((0,), dtype=int), np.empty((0,), dtype=int)
    gid = 0

    # build target, group ID vectors.
    relvalues = range(gmax + 1)

    # build y/target and group-id vectors with user-specified group sizes.
    if group is not None and hasattr(group, '__len__'):
        n_samples = np.sum(group)

        for i, gsize in enumerate(group):
            y_vec = np.concatenate((y_vec, rnd_generator.choice(relvalues, size=gsize, replace=True)))
            group_id_vec = np.concatenate((group_id_vec, [i] * gsize))

    # build y/target and group-id vectors according to n_samples, avg_gs, and random_gs.
    else:
        while len(y_vec) < n_samples:
            gsize = avg_gs if not random_gs else rnd_generator.poisson(avg_gs)

            # groups should contain > 1 element for pairwise learning objective.
            if gsize < 1:
                continue

            y_vec = np.append(y_vec, rnd_generator.choice(relvalues, size=gsize, replace=True))
            group_id_vec = np.append(group_id_vec, [gid] * gsize)
            gid += 1

        y_vec, group_id_vec = y_vec[0:n_samples], group_id_vec[0:n_samples]

    # build feature data, X. Transform first few into informative features.
    n_informative = max(min(n_features, n_informative), 0)
    X = rnd_generator.uniform(size=(n_samples, n_features))

    for j in range(n_informative):
        bias, coef = rnd_generator.normal(size=2)
        X[:, j] = bias + coef * y_vec

    return X, y_vec, group_id_vec


def _create_ranking_data(n_samples=100, output='array', chunk_size=50, **kwargs):
    X, y, g = _make_ranking(n_samples=n_samples, random_state=42, **kwargs)
    rnd = np.random.RandomState(42)
    w = rnd.rand(X.shape[0]) * 0.01
    g_rle = np.array([sum([1 for _ in grp]) for _, grp in itertools.groupby(g)])

    if output == 'dataframe':

        # add target, weight, and group to DataFrame so that partitions abide by group boundaries.
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        X = X_df.copy()
        X_df = X_df.assign(y=y, g=g, w=w)

        # set_index ensures partitions are based on group id. See https://bit.ly/3pAWyNw.
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
        dX, dy, dw, dg = list(), list(), list(), list()
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
        raise ValueError('ranking data creation only supported for Dask arrays and dataframes')

    return X, y, w, g_rle, dX, dy, dw, dg


def _create_data(objective, n_samples=100, centers=2, output='array', chunk_size=50):
    if objective == 'classification':
        X, y = make_blobs(n_samples=n_samples, centers=centers, random_state=42)
    elif objective == 'regression':
        X, y = make_regression(n_samples=n_samples, random_state=42)
    else:
        raise ValueError(objective)
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
        dX = da.from_array(X, chunks=(chunk_size, X.shape[1])).map_blocks(scipy.sparse.csr_matrix)
        dy = da.from_array(y, chunks=chunk_size)
        dw = da.from_array(weights, chunk_size)
    else:
        raise ValueError("Unknown output type %s" % output)

    return X, y, weights, dX, dy, dw


@pytest.mark.parametrize('output', data_output)
@pytest.mark.parametrize('centers', data_centers)
def test_classifier(output, centers, client, listen_port):
    X, y, w, dX, dy, dw = _create_data('classification', output=output, centers=centers)

    dask_classifier = dlgbm.DaskLGBMClassifier(
        time_out=5,
        local_listen_port=listen_port,
        n_estimators=10,
        num_leaves=10
    )
    dask_classifier = dask_classifier.fit(dX, dy, sample_weight=dw, client=client)
    p1 = dask_classifier.predict(dX)
    p1_proba = dask_classifier.predict_proba(dX).compute()
    s1 = accuracy_score(dy, p1)
    p1 = p1.compute()

    local_classifier = lightgbm.LGBMClassifier(n_estimators=10, num_leaves=10)
    local_classifier.fit(X, y, sample_weight=w)
    p2 = local_classifier.predict(X)
    p2_proba = local_classifier.predict_proba(X)
    s2 = local_classifier.score(X, y)

    assert_eq(s1, s2)
    assert_eq(p1, p2)
    assert_eq(y, p1)
    assert_eq(y, p2)
    assert_eq(p1_proba, p2_proba, atol=0.3)

    client.close()


def test_training_does_not_fail_on_port_conflicts(client):
    _, _, _, dX, dy, dw = _create_data('classification', output='array')

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 12400))

        dask_classifier = dlgbm.DaskLGBMClassifier(
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

    client.close()


@pytest.mark.parametrize('output', data_output)
@pytest.mark.parametrize('centers', data_centers)
def test_classifier_proba(output, centers, client, listen_port):
    X, y, w, dX, dy, dw = _create_data('classification', output=output, centers=centers)

    dask_classifier = dlgbm.DaskLGBMClassifier(time_out=5, local_listen_port=listen_port)
    dask_classifier = dask_classifier.fit(dX, dy, sample_weight=dw, client=client)
    p1 = dask_classifier.predict_proba(dX)
    p1 = p1.compute()

    local_classifier = lightgbm.LGBMClassifier()
    local_classifier.fit(X, y, sample_weight=w)
    p2 = local_classifier.predict_proba(X)

    assert_eq(p1, p2, atol=0.3)

    client.close()


def test_classifier_local_predict(client, listen_port):
    X, y, w, dX, dy, dw = _create_data('classification', output='array')

    dask_classifier = dlgbm.DaskLGBMClassifier(
        time_out=5,
        local_listen_port=listen_port,
        n_estimators=10,
        num_leaves=10
    )
    dask_classifier = dask_classifier.fit(dX, dy, sample_weight=dw, client=client)
    p1 = dask_classifier.to_local().predict(dX)

    local_classifier = lightgbm.LGBMClassifier(n_estimators=10, num_leaves=10)
    local_classifier.fit(X, y, sample_weight=w)
    p2 = local_classifier.predict(X)

    assert_eq(p1, p2)
    assert_eq(y, p1)
    assert_eq(y, p2)

    client.close()


@pytest.mark.parametrize('output', data_output)
def test_regressor(output, client, listen_port):
    X, y, w, dX, dy, dw = _create_data('regression', output=output)

    dask_regressor = dlgbm.DaskLGBMRegressor(
        time_out=5,
        local_listen_port=listen_port,
        seed=42,
        num_leaves=10
    )
    dask_regressor = dask_regressor.fit(dX, dy, client=client, sample_weight=dw)
    p1 = dask_regressor.predict(dX)
    if output != 'dataframe':
        s1 = r2_score(dy, p1)
    p1 = p1.compute()

    local_regressor = lightgbm.LGBMRegressor(seed=42, num_leaves=10)
    local_regressor.fit(X, y, sample_weight=w)
    s2 = local_regressor.score(X, y)
    p2 = local_regressor.predict(X)

    # Scores should be the same
    if output != 'dataframe':
        assert_eq(s1, s2, atol=.01)

    # Predictions should be roughly the same
    assert_eq(y, p1, rtol=1., atol=100.)
    assert_eq(y, p2, rtol=1., atol=50.)

    client.close()


@pytest.mark.parametrize('output', data_output)
@pytest.mark.parametrize('alpha', [.1, .5, .9])
def test_regressor_quantile(output, client, listen_port, alpha):
    X, y, w, dX, dy, dw = _create_data('regression', output=output)

    dask_regressor = dlgbm.DaskLGBMRegressor(
        local_listen_port=listen_port,
        seed=42,
        objective='quantile',
        alpha=alpha,
        n_estimators=10,
        num_leaves=10
    )
    dask_regressor = dask_regressor.fit(dX, dy, client=client, sample_weight=dw)
    p1 = dask_regressor.predict(dX).compute()
    q1 = np.count_nonzero(y < p1) / y.shape[0]

    local_regressor = lightgbm.LGBMRegressor(
        seed=42,
        objective='quantile',
        alpha=alpha,
        n_estimatores=10,
        num_leaves=10
    )
    local_regressor.fit(X, y, sample_weight=w)
    p2 = local_regressor.predict(X)
    q2 = np.count_nonzero(y < p2) / y.shape[0]

    # Quantiles should be right
    np.testing.assert_allclose(q1, alpha, atol=0.2)
    np.testing.assert_allclose(q2, alpha, atol=0.2)

    client.close()


def test_regressor_local_predict(client, listen_port):
    X, y, _, dX, dy, dw = _create_data('regression', output='array')

    dask_regressor = dlgbm.DaskLGBMRegressor(
        local_listen_port=listen_port,
        seed=42,
        n_estimators=10,
        num_leaves=10
    )
    dask_regressor = dask_regressor.fit(dX, dy, sample_weight=dw, client=client)
    p1 = dask_regressor.predict(dX)
    p2 = dask_regressor.to_local().predict(X)
    s1 = r2_score(dy, p1)
    p1 = p1.compute()
    s2 = dask_regressor.to_local().score(X, y)

    # Predictions and scores should be the same
    assert_eq(s1, s2)
    assert_eq(p1, p2)

    client.close()


@pytest.mark.parametrize('output', ['array', 'dataframe'])
@pytest.mark.parametrize('group', [None, group_sizes])
def test_ranker(output, client, listen_port, group):

    if os.getenv('TASK', '') == 'gpu':
        pytest.skip('Ranker fails to run with GPU interface')

    X, y, w, g, dX, dy, dw, dg = _create_ranking_data(output=output, group=group)

    # -- use many trees + leaves to overfit, help ensure that dask data-parallel strategy matches that of
    # -- serial learner. See https://github.com/microsoft/LightGBM/issues/3292#issuecomment-671288210.
    dask_ranker = dlgbm.DaskLGBMRanker(time_out=5, local_listen_port=listen_port,
                                       n_estimators=50, num_leaves=20, seed=42, min_child_samples=1)
    dask_ranker = dask_ranker.fit(dX, dy, sample_weight=dw, group=dg, client=client)
    rnkvec_dask = dask_ranker.predict(dX)
    rnkvec_dask = rnkvec_dask.compute()

    local_ranker = lightgbm.LGBMRanker(n_estimators=50, num_leaves=20, seed=42, min_child_samples=1)
    local_ranker.fit(X, y, sample_weight=w, group=g)
    rnkvec_local = local_ranker.predict(X)

    # distributed ranker should be able to rank decently well and should
    # have high rank correlation with scores from serial ranker.
    dcor = spearmanr(rnkvec_dask, y).correlation
    assert dcor > 0.6
    assert spearmanr(rnkvec_dask, rnkvec_local).correlation > 0.9

    client.close()


@pytest.mark.parametrize('output', ['array', 'dataframe'])
@pytest.mark.parametrize('group', [None, group_sizes])
def test_ranker_local_predict(output, client, listen_port, group):

    if os.getenv('TASK', '') == 'gpu':
        pytest.skip('Ranker fails to run with GPU interface')

    X, y, w, g, dX, dy, dw, dg = _create_ranking_data(output=output, group=group)

    dask_ranker = dlgbm.DaskLGBMRanker(time_out=5, local_listen_port=listen_port,
                                       n_estimators=10, num_leaves=10, seed=42, min_child_samples=1)
    dask_ranker = dask_ranker.fit(dX, dy, group=dg, client=client)
    rnkvec_dask = dask_ranker.predict(dX)
    rnkvec_dask = rnkvec_dask.compute()
    rnkvec_local = dask_ranker.to_local().predict(X)

    # distributed and to-local scores should be the same.
    assert_eq(rnkvec_dask, rnkvec_local)

    client.close()


def test_find_open_port_works():
    worker_ip = '127.0.0.1'
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((worker_ip, 12400))
        new_port = dlgbm._find_open_port(
            worker_ip=worker_ip,
            local_listen_port=12400,
            ports_to_skip=set()
        )
        assert new_port == 12401

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s_1:
        s_1.bind((worker_ip, 12400))
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s_2:
            s_2.bind((worker_ip, 12401))
            new_port = dlgbm._find_open_port(
                worker_ip=worker_ip,
                local_listen_port=12400,
                ports_to_skip=set()
            )
            assert new_port == 12402


@gen_cluster(client=True, timeout=None)
def test_errors(c, s, a, b):
    def f(part):
        raise Exception('foo')

    df = dd.demo.make_timeseries()
    df = df.map_partitions(f, meta=df._meta)
    with pytest.raises(Exception) as info:
        yield dlgbm._train(c, df, df.x, params={}, model_factory=lightgbm.LGBMClassifier)
        assert 'foo' in str(info.value)
