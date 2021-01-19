# coding: utf-8
"""Distributed training with LightGBM and Dask.distributed.

This module enables you to perform distributed training with LightGBM on Dask.Array and Dask.DataFrame collections.
It is based on dask-xgboost package.
"""
import logging
import socket
from collections import defaultdict
from typing import Dict, Iterable
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from dask import array as da
from dask import dataframe as dd
from dask import delayed
from dask.distributed import Client, default_client, get_worker, wait

from .basic import _ConfigAliases, _LIB, _safe_call
from .sklearn import LGBMClassifier, LGBMRegressor

import scipy.sparse as ss

logger = logging.getLogger(__name__)


def _find_open_port(worker_ip: str, local_listen_port: int, ports_to_skip: Iterable[int]) -> int:
    """Find an open port.

    This function tries to find a free port on the machine it's run on. It is intended to
    be run once on each Dask worker, sequentially.

    Parameters
    ----------
    worker_ip : str
        IP address for the Dask worker.
    local_listen_port : int
        First port to try when searching for open ports.
    ports_to_skip: Iterable[int]
        An iterable of integers referring to ports that should be skipped. Since multiple Dask
        workers can run on the same physical machine, this method may be called multiple times
        on the same machine. ``ports_to_skip`` is used to ensure that LightGBM doesn't try to use
        the same port for two worker processes running on the same machine.

    Returns
    -------
    result : int
        A free port on the machine referenced by ``worker_ip``.
    """
    max_tries = 1000
    out_port = None
    found_port = False
    for i in range(max_tries):
        out_port = local_listen_port + i
        if out_port in ports_to_skip:
            continue
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((worker_ip, out_port))
            found_port = True
            break
        # if unavailable, you'll get OSError: Address already in use
        except OSError:
            continue
    if not found_port:
        msg = "LightGBM tried %s:%d-%d and could not create a connection. Try setting local_listen_port to a different value."
        raise RuntimeError(msg % (worker_ip, local_listen_port, out_port))
    return out_port


def _find_ports_for_workers(client: Client, worker_addresses: Iterable[str], local_listen_port: int) -> Dict[str, int]:
    """Find an open port on each worker.

    LightGBM distributed training uses TCP sockets by default, and this method is used to
    identify open ports on each worker so LightGBM can reliable create those sockets.

    Parameters
    ----------
    client : dask.distributed.Client
        Dask client.
    worker_addresses : Iterable[str]
        An iterable of addresses for workers in the cluster. These are strings of the form ``<protocol>://<host>:port``
    local_listen_port : int
        First port to try when searching for open ports.

    Returns
    -------
    result : Dict[str, int]
        Dictionary where keys are worker addresses and values are an open port for LightGBM to use.
    """
    lightgbm_ports = set()
    worker_ip_to_port = {}
    for worker_address in worker_addresses:
        port = client.submit(
            func=_find_open_port,
            workers=[worker_address],
            worker_ip=urlparse(worker_address).hostname,
            local_listen_port=local_listen_port,
            ports_to_skip=lightgbm_ports
        ).result()
        lightgbm_ports.add(port)
        worker_ip_to_port[worker_address] = port

    return worker_ip_to_port


def _concat(seq):
    if isinstance(seq[0], np.ndarray):
        return np.concatenate(seq, axis=0)
    elif isinstance(seq[0], (pd.DataFrame, pd.Series)):
        return pd.concat(seq, axis=0)
    elif isinstance(seq[0], ss.spmatrix):
        return ss.vstack(seq, format='csr')
    else:
        raise TypeError('Data must be one of: numpy arrays, pandas dataframes, sparse matrices (from scipy). Got %s.' % str(type(seq[0])))


def _train_part(params, model_factory, list_of_parts, worker_address_to_port, return_model,
                time_out=120, **kwargs):
    local_worker_address = get_worker().address
    machine_list = ','.join([
        '%s:%d' % (urlparse(worker_address).hostname, port)
        for worker_address, port
        in worker_address_to_port.items()
    ])
    network_params = {
        'machines': machine_list,
        'local_listen_port': worker_address_to_port[local_worker_address],
        'time_out': time_out,
        'num_machines': len(worker_address_to_port)
    }
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

    tree_learner = None
    for tree_learner_param in _ConfigAliases.get('tree_learner'):
        tree_learner = params.get(tree_learner_param)
        if tree_learner is not None:
            break

    allowed_tree_learners = {
        'data',
        'data_parallel',
        'feature',
        'feature_parallel',
        'voting',
        'voting_parallel'
    }
    if tree_learner is None:
        logger.warning('Parameter tree_learner not set. Using "data" as default"')
        params['tree_learner'] = 'data'
    elif tree_learner.lower() not in allowed_tree_learners:
        logger.warning('Parameter tree_learner set to %s, which is not allowed. Using "data" as default' % tree_learner)
        params['tree_learner'] = 'data'

    local_listen_port = 12400
    for port_param in _ConfigAliases.get('local_listen_port'):
        val = params.get(port_param)
        if val is not None:
            local_listen_port = val
            break

    # find an open port on each worker. note that multiple workers can run
    # on the same machine, so this needs to ensure that each one gets its
    # own port
    worker_address_to_port = _find_ports_for_workers(
        client=client,
        worker_addresses=worker_map.keys(),
        local_listen_port=local_listen_port
    )

    # Tell each worker to train on the parts that it has locally
    futures_classifiers = [client.submit(_train_part,
                                         model_factory=model_factory,
                                         params={**params, 'num_threads': worker_ncores[worker]},
                                         list_of_parts=list_of_parts,
                                         worker_address_to_port=worker_address_to_port,
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
