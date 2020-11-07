import dask.array as da
import dask.dataframe as dd
import lightgbm
import numpy as np
import pandas as pd
import pytest
import scipy.sparse
from dask.array.utils import assert_eq
from dask_ml.metrics import accuracy_score, r2_score

from distributed.utils_test import client, cluster_fixture, loop, gen_cluster  # noqa
from sklearn.datasets import make_blobs, make_regression
from sklearn.metrics import confusion_matrix

import lightgbm.dask as dlgbm

data_output = ['array', 'scipy_csr_matrix', 'dataframe']
data_centers = [[[-4, -4], [4, 4]], [[-4, -4], [4, 4], [-4, 4]]]


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
    w = rnd.rand(X.shape[0]) * 0.01

    if output == 'array':
        dX = da.from_array(X, (chunk_size, X.shape[1]))
        dy = da.from_array(y, chunk_size)
        dw = da.from_array(w, chunk_size)
    elif output == 'dataframe':
        X_df = pd.DataFrame(X, columns=['feature_%d' % i for i in range(X.shape[1])])
        y_df = pd.Series(y, name='target')
        dX = dd.from_pandas(X_df, chunksize=chunk_size)
        dy = dd.from_pandas(y_df, chunksize=chunk_size)
        dw = dd.from_array(w, chunksize=chunk_size)
    elif output == 'scipy_csr_matrix':
        dX = da.from_array(X, chunks=(chunk_size, X.shape[1])).map_blocks(scipy.sparse.csr_matrix)
        dy = da.from_array(y, chunks=chunk_size)
        dw = da.from_array(w, chunk_size)

    return X, y, w, dX, dy, dw


@pytest.mark.parametrize('output', data_output)
@pytest.mark.parametrize('centers', data_centers)
def test_classifier(output, centers, client, listen_port):  # noqa
    X, y, w, dX, dy, dw = _create_data('classification', output=output, centers=centers)

    a = dlgbm.LGBMClassifier(time_out=5, local_listen_port=listen_port)
    a = a.fit(dX, dy, sample_weight=dw, client=client)
    p1 = a.predict(dX, client=client)
    s1 = accuracy_score(dy, p1)
    p1 = p1.compute()

    b = lightgbm.LGBMClassifier()
    b.fit(X, y, sample_weight=w)
    p2 = b.predict(X)
    s2 = b.score(X, y)
    print(confusion_matrix(y, p1))
    print(confusion_matrix(y, p2))

    assert_eq(s1, s2)
    print(s1)

    assert_eq(p1, p2)
    assert_eq(y, p1)
    assert_eq(y, p2)


@pytest.mark.parametrize('output', data_output)
@pytest.mark.parametrize('centers', data_centers)
def test_classifier_proba(output, centers, client, listen_port):  # noqa
    X, y, w, dX, dy, dw = _create_data('classification', output=output, centers=centers)

    a = dlgbm.LGBMClassifier(time_out=5, local_listen_port=listen_port)
    a = a.fit(dX, dy, sample_weight=dw, client=client)
    p1 = a.predict_proba(dX, client=client)
    p1 = p1.compute()

    b = lightgbm.LGBMClassifier()
    b.fit(X, y, sample_weight=w)
    p2 = b.predict_proba(X)

    assert_eq(p1, p2, atol=0.3)


def test_classifier_local_predict(client, listen_port):  # noqa
    X, y, w, dX, dy, dw = _create_data('classification', output='array')

    a = dlgbm.LGBMClassifier(time_out=5, local_listen_port=listen_port)
    a = a.fit(dX, dy, sample_weight=dw, client=client)
    p1 = a.to_local().predict(dX)

    b = lightgbm.LGBMClassifier()
    b.fit(X, y, sample_weight=w)
    p2 = b.predict(X)

    assert_eq(p1, p2)
    assert_eq(y, p1)
    assert_eq(y, p2)


@pytest.mark.parametrize('output', data_output)
def test_regressor(output, client, listen_port):  # noqa
    X, y, w, dX, dy, dw = _create_data('regression', output=output)

    a = dlgbm.LGBMRegressor(time_out=5, local_listen_port=listen_port, seed=42)
    a = a.fit(dX, dy, client=client, sample_weight=dw)
    p1 = a.predict(dX, client=client)
    if output != 'dataframe':
        s1 = r2_score(dy, p1)
    p1 = p1.compute()

    b = lightgbm.LGBMRegressor(seed=42)
    b.fit(X, y, sample_weight=w)
    s2 = b.score(X, y)
    p2 = b.predict(X)

    # Scores should be the same
    if output != 'dataframe':
        assert_eq(s1, s2, atol=.01)

    # Predictions should be roughly the same
    assert_eq(y, p1, rtol=1., atol=100.)
    assert_eq(y, p2, rtol=1., atol=50.)


@pytest.mark.parametrize('output', data_output)
@pytest.mark.parametrize('alpha', [.1, .5, .9])
def test_regressor_quantile(output, client, listen_port, alpha):  # noqa
    X, y, w, dX, dy, dw = _create_data('regression', output=output)

    a = dlgbm.LGBMRegressor(local_listen_port=listen_port, seed=42, objective='quantile', alpha=alpha)
    a = a.fit(dX, dy, client=client, sample_weight=dw)
    p1 = a.predict(dX, client=client).compute()
    q1 = np.count_nonzero(y < p1) / y.shape[0]

    b = lightgbm.LGBMRegressor(seed=42, objective='quantile', alpha=alpha)
    b.fit(X, y, sample_weight=w)
    p2 = b.predict(X)
    q2 = np.count_nonzero(y < p2) / y.shape[0]

    # Quantiles should be right
    np.isclose(q1, alpha, atol=.1)
    np.isclose(q2, alpha, atol=.1)


def test_regressor_local_predict(client, listen_port):  # noqa
    X, y, w, dX, dy, dw = _create_data('regression', output='array')

    a = dlgbm.LGBMRegressor(local_listen_port=listen_port, seed=42)
    a = a.fit(dX, dy, sample_weight=dw, client=client)
    p1 = a.predict(dX)
    p2 = a.to_local().predict(X)
    s1 = r2_score(dy, p1)
    p1 = p1.compute()
    s2 = a.to_local().score(X, y)
    print(s1)

    # Predictions and scores should be the same
    assert_eq(p1, p2)
    np.isclose(s1, s2)


def test_build_network_params():
    workers_ips = [
        'tcp://192.168.0.1:34545',
        'tcp://192.168.0.2:34346',
        'tcp://192.168.0.3:34347'
    ]

    params = dlgbm.build_network_params(workers_ips, 'tcp://192.168.0.2:34346', 12400, 120)
    exp_params = {
        'machines': '192.168.0.1:12400,192.168.0.2:12401,192.168.0.3:12402',
        'local_listen_port': 12401,
        'num_machines': len(workers_ips),
        'time_out': 120
    }
    assert exp_params == params


@gen_cluster(client=True, timeout=None)
def test_errors(c, s, a, b):
    def f(part):
        raise Exception('foo')

    df = dd.demo.make_timeseries()
    df = df.map_partitions(f, meta=df._meta)
    with pytest.raises(Exception) as info:
        yield dlgbm.train(c, df, df.x, params={}, model_factory=lightgbm.LGBMClassifier)
        assert 'foo' in str(info.value)
