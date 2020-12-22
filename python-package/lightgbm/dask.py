# coding: utf-8
"""Distributed training with LightGBM and Dask.distributed.

This module enables you to perform distributed training with LightGBM on Dask.Array and Dask.DataFrame collections.
It is based on dask-xgboost package.
"""
import logging
from collections import defaultdict
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from dask import array as da
from dask import dataframe as dd
from dask import delayed
from dask.distributed import default_client, get_worker, wait

from .basic import _LIB, _safe_call
from .sklearn import LGBMClassifier, LGBMRegressor

import scipy.sparse as ss

logger = logging.getLogger(__name__)


def _parse_host_port(address):
    parsed = urlparse(address)
    return parsed.hostname, parsed.port


def _build_network_params(worker_addresses, local_worker_ip, local_listen_port, time_out):
    """Build network parameters suitable for LightGBM C backend.

    Parameters
    ----------
    worker_addresses : iterable of str - collection of worker addresses in `<protocol>://<host>:port` format
    local_worker_ip : str
    local_listen_port : int
    time_out : int

    Returns
    -------
    params: dict
    """
    addr_port_map = {addr: (local_listen_port + i) for i, addr in enumerate(worker_addresses)}
    params = {
        'machines': ','.join('%s:%d' % (_parse_host_port(addr)[0], port) for addr, port in addr_port_map.items()),
        'local_listen_port': addr_port_map[local_worker_ip],
        'time_out': time_out,
        'num_machines': len(addr_port_map)
    }
    return params


def _concat(seq):
    if isinstance(seq[0], np.ndarray):
        return np.concatenate(seq, axis=0)
    elif isinstance(seq[0], (pd.DataFrame, pd.Series)):
        return pd.concat(seq, axis=0)
    elif isinstance(seq[0], ss.spmatrix):
        return ss.vstack(seq, format='csr')
    else:
        raise TypeError('Data must be one of: numpy arrays, pandas dataframes, sparse matrices (from scipy). Got %s.' % str(type(seq[0])))


def _train_part(params, model_factory, list_of_parts, worker_addresses, return_model, local_listen_port=12400,
                time_out=120, **kwargs):
    network_params = _build_network_params(worker_addresses, get_worker().address, local_listen_port, time_out)
    params.update(network_params)

    # Concatenate many parts into one
    parts = tuple(zip(*list_of_parts))
    data = _concat(parts[0])
    label = _concat(parts[1])
    weight = _concat(parts[2]) if len(parts) == 3 else None

    try:
        model = model_factory(**params)
        model.fit(data, label, sample_weight=weight, **kwargs)
    finally:
        _safe_call(_LIB.LGBM_NetworkFree())

    return model if return_model else None


def _split_to_parts(data, is_matrix):
    parts = data.to_delayed()
    if isinstance(parts, np.ndarray):
        assert (parts.shape[1] == 1) if is_matrix else (parts.ndim == 1 or parts.shape[1] == 1)
        parts = parts.flatten().tolist()
    return parts


def _train(client, data, label, params, model_factory, weight=None, **kwargs):
    """Inner train routine.

    Parameters
    ----------
    client: dask.Client - client
    X : dask array of shape = [n_samples, n_features]
        Input feature matrix.
    y : dask array of shape = [n_samples]
        The target values (class labels in classification, real numbers in regression).
    params : dict
    model_factory : lightgbm.LGBMClassifier or lightgbm.LGBMRegressor class
    sample_weight : array-like of shape = [n_samples] or None, optional (default=None)
            Weights of training data.
    """
    # Split arrays/dataframes into parts. Arrange parts into tuples to enforce co-locality
    data_parts = _split_to_parts(data, is_matrix=True)
    label_parts = _split_to_parts(label, is_matrix=False)
    if weight is None:
        parts = list(map(delayed, zip(data_parts, label_parts)))
    else:
        weight_parts = _split_to_parts(weight, is_matrix=False)
        parts = list(map(delayed, zip(data_parts, label_parts, weight_parts)))

    # Start computation in the background
    parts = client.compute(parts)
    wait(parts)

    for part in parts:
        if part.status == 'error':
            return part  # trigger error locally

    # Find locations of all parts and map them to particular Dask workers
    key_to_part_dict = dict([(part.key, part) for part in parts])
    who_has = client.who_has(parts)
    worker_map = defaultdict(list)
    for key, workers in who_has.items():
        worker_map[next(iter(workers))].append(key_to_part_dict[key])

    master_worker = next(iter(worker_map))
    worker_ncores = client.ncores()

    if 'tree_learner' not in params or params['tree_learner'].lower() not in {'data', 'feature', 'voting'}:
        logger.warning('Parameter tree_learner not set or set to incorrect value '
                       '(%s), using "data" as default', params.get("tree_learner", None))
        params['tree_learner'] = 'data'

    # Tell each worker to train on the parts that it has locally
    futures_classifiers = [client.submit(_train_part,
                                         model_factory=model_factory,
                                         params={**params, 'num_threads': worker_ncores[worker]},
                                         list_of_parts=list_of_parts,
                                         worker_addresses=list(worker_map.keys()),
                                         local_listen_port=params.get('local_listen_port', 12400),
                                         time_out=params.get('time_out', 120),
                                         return_model=(worker == master_worker),
                                         **kwargs)
                           for worker, list_of_parts in worker_map.items()]

    results = client.gather(futures_classifiers)
    results = [v for v in results if v]
    return results[0]


def _predict_part(part, model, proba, **kwargs):
    data = part.values if isinstance(part, pd.DataFrame) else part

    if data.shape[0] == 0:
        result = np.array([])
    elif proba:
        result = model.predict_proba(data, **kwargs)
    else:
        result = model.predict(data, **kwargs)

    if isinstance(part, pd.DataFrame):
        if proba:
            result = pd.DataFrame(result, index=part.index)
        else:
            result = pd.Series(result, index=part.index, name='predictions')

    return result


def _predict(model, data, proba=False, dtype=np.float32, **kwargs):
    """Inner predict routine.

    Parameters
    ----------
    model :
    data : dask array of shape = [n_samples, n_features]
        Input feature matrix.
    proba : bool
        Should method return results of predict_proba (proba == True) or predict (proba == False)
    dtype : np.dtype
        Dtype of the output
    kwargs : other parameters passed to predict or predict_proba method
    """
    if isinstance(data, dd._Frame):
        return data.map_partitions(_predict_part, model=model, proba=proba, **kwargs).values
    elif isinstance(data, da.Array):
        if proba:
            kwargs['chunks'] = (data.chunks[0], (model.n_classes_,))
        else:
            kwargs['drop_axis'] = 1
        return data.map_blocks(_predict_part, model=model, proba=proba, dtype=dtype, **kwargs)
    else:
        raise TypeError('Data must be either Dask array or dataframe. Got %s.' % str(type(data)))


class _LGBMModel:

    def _fit(self, model_factory, X, y=None, sample_weight=None, client=None, **kwargs):
        """Docstring is inherited from the LGBMModel."""
        if client is None:
            client = default_client()

        params = self.get_params(True)
        model = _train(client, X, y, params, model_factory, sample_weight, **kwargs)

        self.set_params(**model.get_params())
        self._copy_extra_params(model, self)

        return self

    def _to_local(self, model_factory):
        model = model_factory(**self.get_params())
        self._copy_extra_params(self, model)
        return model

    @staticmethod
    def _copy_extra_params(source, dest):
        params = source.get_params()
        attributes = source.__dict__
        extra_param_names = set(attributes.keys()).difference(params.keys())
        for name in extra_param_names:
            setattr(dest, name, attributes[name])


class DaskLGBMClassifier(_LGBMModel, LGBMClassifier):
    """Distributed version of lightgbm.LGBMClassifier."""

    def fit(self, X, y=None, sample_weight=None, client=None, **kwargs):
        """Docstring is inherited from the LGBMModel."""
        return self._fit(LGBMClassifier, X, y, sample_weight, client, **kwargs)
    fit.__doc__ = LGBMClassifier.fit.__doc__

    def predict(self, X, **kwargs):
        """Docstring is inherited from the lightgbm.LGBMClassifier.predict."""
        return _predict(self.to_local(), X, dtype=self.classes_.dtype, **kwargs)
    predict.__doc__ = LGBMClassifier.predict.__doc__

    def predict_proba(self, X, **kwargs):
        """Docstring is inherited from the lightgbm.LGBMClassifier.predict_proba."""
        return _predict(self.to_local(), X, proba=True, **kwargs)
    predict_proba.__doc__ = LGBMClassifier.predict_proba.__doc__

    def to_local(self):
        """Create regular version of lightgbm.LGBMClassifier from the distributed version.

        Returns
        -------
        model : lightgbm.LGBMClassifier
        """
        return self._to_local(LGBMClassifier)


class DaskLGBMRegressor(_LGBMModel, LGBMRegressor):
    """Docstring is inherited from the lightgbm.LGBMRegressor."""

    def fit(self, X, y=None, sample_weight=None, client=None, **kwargs):
        """Docstring is inherited from the lightgbm.LGBMRegressor.fit."""
        return self._fit(LGBMRegressor, X, y, sample_weight, client, **kwargs)
    fit.__doc__ = LGBMRegressor.fit.__doc__

    def predict(self, X, **kwargs):
        """Docstring is inherited from the lightgbm.LGBMRegressor.predict."""
        return _predict(self.to_local(), X, **kwargs)
    predict.__doc__ = LGBMRegressor.predict.__doc__

    def to_local(self):
        """Create regular version of lightgbm.LGBMRegressor from the distributed version.

        Returns
        -------
        model : lightgbm.LGBMRegressor
        """
        return self._to_local(LGBMRegressor)
