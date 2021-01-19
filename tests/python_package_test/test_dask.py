# coding: utf-8
import os
import socket
import sys

import pytest
if not sys.platform.startswith("linux"):
    pytest.skip("lightgbm.dask is currently supported in Linux environments", allow_module_level=True)

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import scipy.sparse
from dask.array.utils import assert_eq
from dask_ml.metrics import accuracy_score, r2_score
from distributed.utils_test import client, cluster_fixture, gen_cluster, loop
from sklearn.datasets import make_blobs, make_regression

import lightgbm
import lightgbm.dask as dlgbm

data_output = ['array', 'scipy_csr_matrix', 'dataframe']
data_centers = [[[-4, -4], [4, 4]], [[-4, -4], [4, 4], [-4, 4]]]

pytestmark = [
    pytest.mark.skipif(os.getenv("TASK", "") == "mpi", reason="Fails to run with MPI interface")
]


@pytest.fixture()
def listen_port():
    listen_port.port += 10
    return listen_port.port


listen_port.port = 13000


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
    assert_eq(p1, p2)
    assert_eq(s1, s2)


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
