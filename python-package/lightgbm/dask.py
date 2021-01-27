# coding: utf-8
"""Distributed training with LightGBM and Dask.distributed.

This module enables you to perform distributed training with LightGBM on
Dask.Array and Dask.DataFrame collections.

It is based on dask-lightgbm, which was based on dask-xgboost.
"""
import socket
from collections import defaultdict
from copy import deepcopy
from typing import Dict, Iterable
from urllib.parse import urlparse

import numpy as np
import scipy.sparse as ss

from .basic import _choose_param_value, _ConfigAliases, _LIB, _log_warning, _safe_call, LightGBMError
from .compat import (PANDAS_INSTALLED, pd_DataFrame, pd_Series, concat,
                     SKLEARN_INSTALLED,
                     DASK_INSTALLED, dask_Frame, dask_Array, delayed, Client, default_client, get_worker, wait)
from .sklearn import LGBMClassifier, LGBMRegressor, LGBMRanker


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
    port : int
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
        An iterable of addresses for workers in the cluster. These are strings of the form ``<protocol>://<host>:port``.
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
    elif isinstance(seq[0], (pd_DataFrame, pd_Series)):
        return concat(seq, axis=0)
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

    is_ranker = issubclass(model_factory, LGBMRanker)

    # Concatenate many parts into one
    data = _concat([x['data'] for x in list_of_parts])
    label = _concat([x['label'] for x in list_of_parts])

    if 'weight' in list_of_parts[0]:
        weight = _concat([x['weight'] for x in list_of_parts])
    else:
        weight = None

    if 'group' in list_of_parts[0]:
        group = _concat([x['group'] for x in list_of_parts])
    else:
        group = None

    try:
        model = model_factory(**params)
        if is_ranker:
            model.fit(data, label, sample_weight=weight, group=group, **kwargs)
        else:
            model.fit(data, label, sample_weight=weight, **kwargs)

    finally:
        _safe_call(_LIB.LGBM_NetworkFree())

    return model if return_model else None


def _split_to_parts(data, is_matrix):
    parts = data.to_delayed()
    if isinstance(parts, np.ndarray):
        if is_matrix:
            assert parts.shape[1] == 1
        else:
            assert parts.ndim == 1 or parts.shape[1] == 1
        parts = parts.flatten().tolist()
    return parts


def _train(client, data, label, params, model_factory, sample_weight=None, group=None, **kwargs):
    """Inner train routine.

    Parameters
    ----------
    client : dask.distributed.Client
        Dask client.
    data : dask array of shape = [n_samples, n_features]
        Input feature matrix.
    label : dask array of shape = [n_samples]
        The target values (class labels in classification, real numbers in regression).
    params : dict
        Parameters passed to constructor of the local underlying model.
    model_factory : lightgbm.LGBMClassifier, lightgbm.LGBMRegressor, or lightgbm.LGBMRanker class
        Class of the local underlying model.
    sample_weight : array-like of shape = [n_samples] or None, optional (default=None)
        Weights of training data.
    group : array-like or None, optional (default=None)
        Group/query data.
        Only used in the learning-to-rank task.
        sum(group) = n_samples.
        For example, if you have a 100-document dataset with ``group = [10, 20, 40, 10, 10, 10]``, that means that you have 6 groups,
        where the first 10 records are in the first group, records 11-30 are in the second group, records 31-70 are in the third group, etc.
    **kwargs
        Other parameters passed to ``fit`` method of the local underlying model.

    Returns
    -------
    model : lightgbm.LGBMClassifier, lightgbm.LGBMRegressor, or lightgbm.LGBMRanker class
        Returns fitted underlying model.
    """
    params = deepcopy(params)

    params = _choose_param_value(
        main_param_name="local_listen_port",
        params=params,
        default_value=12400
    )

    params = _choose_param_value(
        main_param_name="tree_learner",
        params=params,
        default_value="data"
    )
    allowed_tree_learners = {
        'data',
        'data_parallel',
        'feature',
        'feature_parallel',
        'voting',
        'voting_parallel'
    }
    if params["tree_learner"] not in allowed_tree_learners:
        _log_warning('Parameter tree_learner set to %s, which is not allowed. Using "data" as default' % tree_learner)
        params['tree_learner'] = 'data'

    if params['tree_learner'] not in {'data', 'data_parallel'}:
        _log_warning(
            'Support for tree_learner %s in lightgbm.dask is experimental and may break in a future release. \n'
            'Use "data" for a stable, well-tested interface.' % params['tree_learner']
        )

    # Some passed-in parameters can be removed:
    #   * 'machines': constructed automatically from Dask worker list
    #   * 'num_machines': set automatically from Dask worker list
    #   * 'num_threads': overridden to match nthreads on each Dask process
    for param_alias in _ConfigAliases.get('machines', 'num_machines', 'num_threads'):
        params.pop(param_alias, None)

    # Split arrays/dataframes into parts. Arrange parts into dicts to enforce co-locality
    data_parts = _split_to_parts(data=data, is_matrix=True)
    label_parts = _split_to_parts(data=label, is_matrix=False)
    parts = [{'data': x, 'label': y} for (x, y) in zip(data_parts, label_parts)]

    if sample_weight is not None:
        weight_parts = _split_to_parts(data=sample_weight, is_matrix=False)
        for i in range(len(parts)):
            parts[i]['weight'] = weight_parts[i]

    if group is not None:
        group_parts = _split_to_parts(data=group, is_matrix=False)
        for i in range(len(parts)):
            parts[i]['group'] = group_parts[i]

    # Start computation in the background
    parts = list(map(delayed, parts))
    parts = client.compute(parts)
    wait(parts)

    for part in parts:
        if part.status == 'error':
            return part  # trigger error locally

    # Find locations of all parts and map them to particular Dask workers
    key_to_part_dict = {part.key: part for part in parts}
    who_has = client.who_has(parts)
    worker_map = defaultdict(list)
    for key, workers in who_has.items():
        worker_map[next(iter(workers))].append(key_to_part_dict[key])

    master_worker = next(iter(worker_map))
    worker_ncores = client.ncores()

    # find an open port on each worker. note that multiple workers can run
    # on the same machine, so this needs to ensure that each one gets its
    # own port
    worker_address_to_port = _find_ports_for_workers(
        client=client,
        worker_addresses=worker_map.keys(),
        local_listen_port=params["local_listen_port"]
    )

    # Tell each worker to train on the parts that it has locally
    futures_classifiers = [
        client.submit(
            _train_part,
            model_factory=model_factory,
            params={**params, 'num_threads': worker_ncores[worker]},
            list_of_parts=list_of_parts,
            worker_address_to_port=worker_address_to_port,
            time_out=params.get('time_out', 120),
            return_model=(worker == master_worker),
            **kwargs
        )
        for worker, list_of_parts in worker_map.items()
    ]

    results = client.gather(futures_classifiers)
    results = [v for v in results if v]
    return results[0]


def _predict_part(part, model, raw_score, pred_proba, pred_leaf, pred_contrib, **kwargs):
    data = part.values if isinstance(part, pd_DataFrame) else part

    if data.shape[0] == 0:
        result = np.array([])
    elif pred_proba:
        result = model.predict_proba(
            data,
            raw_score=raw_score,
            pred_leaf=pred_leaf,
            pred_contrib=pred_contrib,
            **kwargs
        )
    else:
        result = model.predict(
            data,
            raw_score=raw_score,
            pred_leaf=pred_leaf,
            pred_contrib=pred_contrib,
            **kwargs
        )

    if isinstance(part, pd_DataFrame):
        if pred_proba or pred_contrib:
            result = pd_DataFrame(result, index=part.index)
        else:
            result = pd_Series(result, index=part.index, name='predictions')

    return result


def _predict(model, data, raw_score=False, pred_proba=False, pred_leaf=False, pred_contrib=False,
             dtype=np.float32, **kwargs):
    """Inner predict routine.

    Parameters
    ----------
    model : lightgbm.LGBMClassifier, lightgbm.LGBMRegressor, or lightgbm.LGBMRanker class
        Fitted underlying model.
    data : dask array of shape = [n_samples, n_features]
        Input feature matrix.
    raw_score : bool, optional (default=False)
        Whether to predict raw scores.
    pred_proba : bool, optional (default=False)
        Should method return results of ``predict_proba`` (``pred_proba=True``) or ``predict`` (``pred_proba=False``).
    pred_leaf : bool, optional (default=False)
        Whether to predict leaf index.
    pred_contrib : bool, optional (default=False)
        Whether to predict feature contributions.
    dtype : np.dtype, optional (default=np.float32)
        Dtype of the output.
    **kwargs
        Other parameters passed to ``predict`` or ``predict_proba`` method.

    Returns
    -------
    predicted_result : dask array of shape = [n_samples] or shape = [n_samples, n_classes]
        The predicted values.
    X_leaves : dask array of shape = [n_samples, n_trees] or shape = [n_samples, n_trees * n_classes]
        If ``pred_leaf=True``, the predicted leaf of every tree for each sample.
    X_SHAP_values : dask array of shape = [n_samples, n_features + 1] or shape = [n_samples, (n_features + 1) * n_classes] or list with n_classes length of such objects
        If ``pred_contrib=True``, the feature contributions for each sample.
    """
    if not all((DASK_INSTALLED, PANDAS_INSTALLED, SKLEARN_INSTALLED)):
        raise LightGBMError('dask, pandas and scikit-learn are required for lightgbm.dask')
    if isinstance(data, dask_Frame):
        return data.map_partitions(
            _predict_part,
            model=model,
            raw_score=raw_score,
            pred_proba=pred_proba,
            pred_leaf=pred_leaf,
            pred_contrib=pred_contrib,
            **kwargs
        ).values
    elif isinstance(data, dask_Array):
        if pred_proba:
            kwargs['chunks'] = (data.chunks[0], (model.n_classes_,))
        else:
            kwargs['drop_axis'] = 1
        return data.map_blocks(
            _predict_part,
            model=model,
            raw_score=raw_score,
            pred_proba=pred_proba,
            pred_leaf=pred_leaf,
            pred_contrib=pred_contrib,
            dtype=dtype,
            **kwargs
        )
    else:
        raise TypeError('Data must be either Dask array or dataframe. Got %s.' % str(type(data)))


class _DaskLGBMModel:
    def _fit(self, model_factory, X, y, sample_weight=None, group=None, client=None, **kwargs):
        if not all((DASK_INSTALLED, PANDAS_INSTALLED, SKLEARN_INSTALLED)):
            raise LightGBMError('dask, pandas and scikit-learn are required for lightgbm.dask')
        if client is None:
            client = default_client()

        params = self.get_params(True)

        model = _train(
            client=client,
            data=X,
            label=y,
            params=params,
            model_factory=model_factory,
            sample_weight=sample_weight,
            group=group,
            **kwargs
        )

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


class DaskLGBMClassifier(LGBMClassifier, _DaskLGBMModel):
    """Distributed version of lightgbm.LGBMClassifier."""

    def fit(self, X, y, sample_weight=None, client=None, **kwargs):
        """Docstring is inherited from the lightgbm.LGBMClassifier.fit."""
        return self._fit(
            model_factory=LGBMClassifier,
            X=X,
            y=y,
            sample_weight=sample_weight,
            client=client,
            **kwargs
        )

    _base_doc = LGBMClassifier.fit.__doc__
    _before_init_score, _init_score, _after_init_score = _base_doc.partition('init_score :')
    fit.__doc__ = (_before_init_score
                   + 'client : dask.distributed.Client or None, optional (default=None)\n'
                   + ' ' * 12 + 'Dask client.\n'
                   + ' ' * 8 + _init_score + _after_init_score)

    def predict(self, X, **kwargs):
        """Docstring is inherited from the lightgbm.LGBMClassifier.predict."""
        return _predict(
            model=self.to_local(),
            data=X,
            dtype=self.classes_.dtype,
            **kwargs
        )

    predict.__doc__ = LGBMClassifier.predict.__doc__

    def predict_proba(self, X, **kwargs):
        """Docstring is inherited from the lightgbm.LGBMClassifier.predict_proba."""
        return _predict(
            model=self.to_local(),
            data=X,
            pred_proba=True,
            **kwargs
        )

    predict_proba.__doc__ = LGBMClassifier.predict_proba.__doc__

    def to_local(self):
        """Create regular version of lightgbm.LGBMClassifier from the distributed version.

        Returns
        -------
        model : lightgbm.LGBMClassifier
            Local underlying model.
        """
        return self._to_local(LGBMClassifier)


class DaskLGBMRegressor(LGBMRegressor, _DaskLGBMModel):
    """Distributed version of lightgbm.LGBMRegressor."""

    def fit(self, X, y, sample_weight=None, client=None, **kwargs):
        """Docstring is inherited from the lightgbm.LGBMRegressor.fit."""
        return self._fit(
            model_factory=LGBMRegressor,
            X=X,
            y=y,
            sample_weight=sample_weight,
            client=client,
            **kwargs
        )

    _base_doc = LGBMRegressor.fit.__doc__
    _before_init_score, _init_score, _after_init_score = _base_doc.partition('init_score :')
    fit.__doc__ = (_before_init_score
                   + 'client : dask.distributed.Client or None, optional (default=None)\n'
                   + ' ' * 12 + 'Dask client.\n'
                   + ' ' * 8 + _init_score + _after_init_score)

    def predict(self, X, **kwargs):
        """Docstring is inherited from the lightgbm.LGBMRegressor.predict."""
        return _predict(
            model=self.to_local(),
            data=X,
            **kwargs
        )

    predict.__doc__ = LGBMRegressor.predict.__doc__

    def to_local(self):
        """Create regular version of lightgbm.LGBMRegressor from the distributed version.

        Returns
        -------
        model : lightgbm.LGBMRegressor
            Local underlying model.
        """
        return self._to_local(LGBMRegressor)


class DaskLGBMRanker(LGBMRanker, _DaskLGBMModel):
    """Distributed version of lightgbm.LGBMRanker."""

    def fit(self, X, y, sample_weight=None, init_score=None, group=None, client=None, **kwargs):
        """Docstring is inherited from the lightgbm.LGBMRanker.fit."""
        if init_score is not None:
            raise RuntimeError('init_score is not currently supported in lightgbm.dask')

        return self._fit(
            model_factory=LGBMRanker,
            X=X,
            y=y,
            sample_weight=sample_weight,
            group=group,
            client=client,
            **kwargs
        )

    _base_doc = LGBMRanker.fit.__doc__
    _before_eval_set, _eval_set, _after_eval_set = _base_doc.partition('eval_set :')
    fit.__doc__ = (_before_eval_set
                   + 'client : dask.distributed.Client or None, optional (default=None)\n'
                   + ' ' * 12 + 'Dask client.\n'
                   + ' ' * 8 + _eval_set + _after_eval_set)

    def predict(self, X, **kwargs):
        """Docstring is inherited from the lightgbm.LGBMRanker.predict."""
        return _predict(self.to_local(), X, **kwargs)

    predict.__doc__ = LGBMRanker.predict.__doc__

    def to_local(self):
        """Create regular version of lightgbm.LGBMRanker from the distributed version.

        Returns
        -------
        model : lightgbm.LGBMRanker
            Local underlying model.
        """
        return self._to_local(LGBMRanker)
