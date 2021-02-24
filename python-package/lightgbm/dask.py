# coding: utf-8
"""Distributed training with LightGBM and dask.distributed.

This module enables you to perform distributed training with LightGBM on
dask.Array and dask.DataFrame collections.

It is based on dask-lightgbm, which was based on dask-xgboost.
"""
import socket
from collections import defaultdict
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Type, Union
from urllib.parse import urlparse

import numpy as np
import scipy.sparse as ss

from .basic import _LIB, LightGBMError, _choose_param_value, _ConfigAliases, _log_info, _log_warning, _safe_call
from .compat import (DASK_INSTALLED, PANDAS_INSTALLED, SKLEARN_INSTALLED, Client, LGBMNotFittedError, concat,
                     dask_Array, dask_DataFrame, dask_Series, default_client, delayed, get_worker, pd_DataFrame,
                     pd_Series, wait)
from .sklearn import LGBMClassifier, LGBMModel, LGBMRanker, LGBMRegressor, _lgbmmodel_doc_fit, _lgbmmodel_doc_predict

_DaskCollection = Union[dask_Array, dask_DataFrame, dask_Series]
_DaskMatrixLike = Union[dask_Array, dask_DataFrame]
_DaskPart = Union[np.ndarray, pd_DataFrame, pd_Series, ss.spmatrix]
_PredictionDtype = Union[Type[np.float32], Type[np.float64], Type[np.int32], Type[np.int64]]


def _get_dask_client(client: Optional[Client]) -> Client:
    """Choose a Dask client to use.

    Parameters
    ----------
    client : dask.distributed.Client or None
        Dask client.

    Returns
    -------
    client : dask.distributed.Client
        A Dask client.
    """
    if client is None:
        return default_client()
    else:
        return client


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
    lightgbm_ports: Set[int] = set()
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


def _concat(seq: List[_DaskPart]) -> _DaskPart:
    if isinstance(seq[0], np.ndarray):
        return np.concatenate(seq, axis=0)
    elif isinstance(seq[0], (pd_DataFrame, pd_Series)):
        return concat(seq, axis=0)
    elif isinstance(seq[0], ss.spmatrix):
        return ss.vstack(seq, format='csr')
    else:
        raise TypeError('Data must be one of: numpy arrays, pandas dataframes, sparse matrices (from scipy). Got %s.' % str(type(seq[0])))


def _train_part(
    params: Dict[str, Any],
    model_factory: Type[LGBMModel],
    list_of_parts: List[Dict[str, _DaskPart]],
    machines: str,
    local_listen_port: int,
    num_machines: int,
    return_model: bool,
    time_out: int = 120,
    evals_provided: bool = False,
    **kwargs: Any
) -> Optional[LGBMModel]:
    network_params = {
        'machines': machines,
        'local_listen_port': local_listen_port,
        'time_out': time_out,
        'num_machines': num_machines
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

    # construct local eval_set data.
    local_eval_set = None
    local_eval_sample_weight = None
    local_eval_group = None
    n_evals = max([len(x.get('eval_set', [])) for x in list_of_parts])
    has_eval_weights = any([x.get('eval_sample_weight') is not None for x in list_of_parts])
    if n_evals:

        local_eval_set = []
        eval_names = []
        if has_eval_weights:
            local_eval_sample_weight = []
        if is_ranker:
            local_eval_group = []

        # consolidate parts of each individual eval component.
        for i in range(n_evals):
            x_e = []
            y_e = []
            w_e = []
            g_e = []
            for part in list_of_parts:

                if not part.get('eval_set'):
                    continue

                # possible that not each part contains parts of each individual (X, y) eval set.
                if i >= len(part['eval_set']):
                    continue

                eval_set = part['eval_set'][i]
                if eval_set == '__train__':
                    x_e.append(part['data'])
                    y_e.append(part['label'])
                else:
                    x, y = eval_set
                    x_e.extend(x)
                    y_e.extend(y)

                eval_weight = part.get('eval_sample_weight')
                if eval_weight:
                    if eval_weight[i] == '__sample_weight__':
                        w_e.append(part['weight'])
                    else:
                        w_e.extend(eval_weight[i])

                eval_group = part.get('eval_group')
                if eval_group:
                    if eval_group[i] == '__group__':
                        g_e.append(part['group'])
                    else:
                        g_e.extend(eval_group[i])

                local_eval_names = part.get('eval_names')
                if local_eval_names:
                    if len(local_eval_names) > len(eval_names):
                        eval_names = local_eval_names

            # _concat each eval component.
            local_eval_set.append((_concat(x_e), _concat(y_e)))
            if w_e:
                local_eval_sample_weight.append(_concat(w_e))
            if g_e:
                local_eval_group.append(_concat(g_e))

    else:
        # when a worker receives no eval_set while other workers have eval data, causes LightGBMExceptions.
        if evals_provided:
            local_worker_address = get_worker().address
            msg = "eval_set was provided but worker %s was not allocated validation data. Try rebalancing data across workers."
            raise RuntimeError(msg % local_worker_address)

    try:
        model = model_factory(**params)
        if is_ranker:
            model.fit(
                data,
                label,
                sample_weight=weight,
                group=group,
                eval_set=local_eval_set,
                eval_sample_weight=local_eval_sample_weight,
                eval_group=local_eval_group,
                eval_names=eval_names if eval_names else None,
                **kwargs
            )
        else:
            if 'eval_at' in kwargs:
                kwargs.pop('eval_at')

            model.fit(
                data,
                label,
                sample_weight=weight,
                eval_set=local_eval_set,
                eval_sample_weight=local_eval_sample_weight,
                eval_names=eval_names if eval_names else None,
                **kwargs
            )

    finally:
        _safe_call(_LIB.LGBM_NetworkFree())

    return model if return_model else None


def _split_to_parts(data: _DaskCollection, is_matrix: bool) -> List[_DaskPart]:
    parts = data.to_delayed()
    if isinstance(parts, np.ndarray):
        if is_matrix:
            assert parts.shape[1] == 1
        else:
            assert parts.ndim == 1 or parts.shape[1] == 1
        parts = parts.flatten().tolist()
    return parts


def _machines_to_worker_map(machines: str, worker_addresses: List[str]) -> Dict[str, int]:
    """Create a worker_map from machines list.

    Given ``machines`` and a list of Dask worker addresses, return a mapping where the keys are
    ``worker_addresses`` and the values are ports from ``machines``.

    Parameters
    ----------
    machines : str
        A comma-delimited list of workers, of the form ``ip1:port,ip2:port``.
    worker_addresses : list of str
        A list of Dask worker addresses, of the form ``{protocol}{hostname}:{port}``, where ``port`` is the port Dask's scheduler uses to talk to that worker.

    Returns
    -------
    result : Dict[str, int]
        Dictionary where keys are work addresses in the form expected by Dask and values are a port for LightGBM to use.
    """
    machine_addresses = machines.split(",")
    machine_to_port = defaultdict(set)
    for address in machine_addresses:
        host, port = address.split(":")
        machine_to_port[host].add(int(port))

    out = {}
    for address in worker_addresses:
        worker_host = urlparse(address).hostname
        out[address] = machine_to_port[worker_host].pop()

    return out


def _train(
    client: Client,
    data: _DaskMatrixLike,
    label: _DaskCollection,
    params: Dict[str, Any],
    model_factory: Type[LGBMModel],
    sample_weight: Optional[_DaskCollection] = None,
    group: Optional[_DaskCollection] = None,
    eval_set: Optional[List[Tuple[_DaskCollection, _DaskCollection]]] = None,
    eval_names: Optional[List[str]] = None,
    eval_sample_weight: Optional[List[_DaskCollection]] = None,
    eval_group: Optional[List[_DaskCollection]] = None,
    **kwargs: Any
) -> LGBMModel:
    """Inner train routine.

    Parameters
    ----------
    client : dask.distributed.Client
        Dask client.
    data : Dask Array or Dask DataFrame of shape = [n_samples, n_features]
        Input feature matrix.
    label : Dask Array, Dask DataFrame or Dask Series of shape = [n_samples]
        The target values (class labels in classification, real numbers in regression).
    params : dict
        Parameters passed to constructor of the local underlying model.
    model_factory : lightgbm.LGBMClassifier, lightgbm.LGBMRegressor, or lightgbm.LGBMRanker class
        Class of the local underlying model.
    sample_weight : Dask Array, Dask DataFrame, Dask Series of shape = [n_samples] or None, optional (default=None)
        Weights of training data.
    group : Dask Array, Dask DataFrame, Dask Series of shape = [n_samples] or None, optional (default=None)
        Group/query data.
        Only used in the learning-to-rank task.
        sum(group) = n_samples.
        For example, if you have a 100-document dataset with ``group = [10, 20, 40, 10, 10, 10]``, that means that you have 6 groups,
        where the first 10 records are in the first group, records 11-30 are in the second group, records 31-70 are in the third group, etc.
    eval_set : List of (X, y) tuples of Dask data collections, or None, optional (default=None)
        List of (X, y) tuple pairs to use as validation sets.
    eval_names: List of strings or None, optional (default=None))
        Names of eval_set.
    eval_sample_weight: List of Dask data collections or None, optional (default=None)
        List of Dask Array or Dask Series, weights for each validation set in eval_set.
    eval_group: List of Dask data collections or None, optional (default=None)
        List of Dask Array or Dask Series, group/query for each validation set in eval_set.
    **kwargs
        Other parameters passed to ``fit`` method of the local underlying model.

    Returns
    -------
    model : lightgbm.LGBMClassifier, lightgbm.LGBMRegressor, or lightgbm.LGBMRanker class
        Returns fitted underlying model.

    Note
    ----

    This method handles setting up the following network parameters based on information
    about the Dask cluster referenced by ``client``.

    * ``local_listen_port``: port that each LightGBM worker opens a listening socket on,
            to accept connections from other workers. This can differ from LightGBM worker
            to LightGBM worker, but does not have to.
    * ``machines``: a comma-delimited list of all workers in the cluster, in the
            form ``ip:port,ip:port``. If running multiple Dask workers on the same host, use different
            ports for each worker. For example, for ``LocalCluster(n_workers=3)``, you might
            pass ``"127.0.0.1:12400,127.0.0.1:12401,127.0.0.1:12402"``.
    * ``num_machines``: number of LightGBM workers.
    * ``timeout``: time in minutes to wait before closing unused sockets.

    The default behavior of this function is to generate ``machines`` from the list of
    Dask workers which hold some piece of the training data, and to search for an open
    port on each worker to be used as ``local_listen_port``.

    If ``machines`` is provided explicitly in ``params``, this function uses the hosts
    and ports in that list directly, and does not do any searching. This means that if
    any of the Dask workers are missing from the list or any of those ports are not free
    when training starts, training will fail.

    If ``local_listen_port`` is provided in ``params`` and ``machines`` is not, this function
    constructs ``machines`` from the list of Dask workers which hold some piece of the
    training data, assuming that each one will use the same ``local_listen_port``.
    """
    params = deepcopy(params)

    # capture whether local_listen_port or its aliases were provided
    listen_port_in_params = any(
        alias in params for alias in _ConfigAliases.get("local_listen_port")
    )

    # capture whether machines or its aliases were provided
    machines_in_params = any(
        alias in params for alias in _ConfigAliases.get("machines")
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
        _log_warning('Parameter tree_learner set to %s, which is not allowed. Using "data" as default' % params['tree_learner'])
        params['tree_learner'] = 'data'

    if params['tree_learner'] not in {'data', 'data_parallel'}:
        _log_warning(
            'Support for tree_learner %s in lightgbm.dask is experimental and may break in a future release. \n'
            'Use "data" for a stable, well-tested interface.' % params['tree_learner']
        )

    # Some passed-in parameters can be removed:
    #   * 'num_machines': set automatically from Dask worker list
    #   * 'num_threads': overridden to match nthreads on each Dask process
    for param_alias in _ConfigAliases.get('num_machines', 'num_threads'):
        if param_alias in params:
            _log_warning(f"Parameter {param_alias} will be ignored.")
            params.pop(param_alias)

    # Split arrays/dataframes into parts. Arrange parts into dicts to enforce co-locality
    data_parts = _split_to_parts(data=data, is_matrix=True)
    label_parts = _split_to_parts(data=label, is_matrix=False)
    parts = [{'data': x, 'label': y} for (x, y) in zip(data_parts, label_parts)]
    n_parts = len(parts)

    if sample_weight is not None:
        weight_parts = _split_to_parts(data=sample_weight, is_matrix=False)
        for i in range(n_parts):
            parts[i]['weight'] = weight_parts[i]

    if group is not None:
        group_parts = _split_to_parts(data=group, is_matrix=False)
        for i in range(n_parts):
            parts[i]['group'] = group_parts[i]

    # evals_set will to be re-constructed into smaller lists of (X, y) tuples, where
    # X and y are each delayed sub-lists of original eval dask Collections.
    if eval_set:
        eval_sets = defaultdict(list)
        if eval_sample_weight:
            eval_sample_weights = defaultdict(list)
        if eval_group:
            eval_groups = defaultdict(list)

        for i, (X, y) in enumerate(eval_set):

            # when individual eval set is equivalent to training data, skip recomputing parts.
            if X is data and y is label:
                for parts_idx in range(n_parts):
                    eval_sets[parts_idx].append('__train__')

            else:
                eval_x_parts = _split_to_parts(data=X, is_matrix=True)
                eval_y_parts = _split_to_parts(data=y, is_matrix=False)

                for j in range(len(eval_x_parts)):
                    parts_idx = j % n_parts

                    x_e = eval_x_parts[j]
                    y_e = eval_y_parts[j]

                    if j < n_parts:
                        eval_sets[parts_idx].append(([x_e], [y_e]))

                    else:
                        eval_sets[parts_idx][-1][0].append(x_e)
                        eval_sets[parts_idx][-1][1].append(y_e)

            if eval_sample_weight:
                if eval_sample_weight[i] is sample_weight:
                    for parts_idx in range(n_parts):
                        eval_sample_weights[parts_idx].append('__sample_weight__')

                else:
                    eval_w_parts = _split_to_parts(data=eval_sample_weight[i], is_matrix=False)

                    # ensure that all evaluation parts map uniquely to one part.
                    for j in range(len(eval_w_parts)):
                        parts_idx = j % n_parts

                        w_e = eval_w_parts[j]

                        if j < n_parts:
                            eval_sample_weights[parts_idx].append([w_e])

                        else:
                            # n_evals = len(eval_sample_weights[parts_idx]) - 1
                            eval_sample_weights[parts_idx][-1].append(w_e)

            if eval_group:
                if eval_group[i] is group:
                    for parts_idx in range(n_parts):
                        eval_groups[parts_idx].append('__group__')

                else:
                    eval_g_parts = _split_to_parts(data=eval_group[i], is_matrix=False)

                    # ensure that all evaluation parts map uniquely to one part.
                    for j in range(len(eval_g_parts)):
                        parts_idx = j % n_parts
                        g_e = eval_g_parts[j]

                        if j < n_parts:
                            eval_groups[parts_idx].append([g_e])

                        else:
                            # n_evals = len(eval_groups[parts_idx]) - 1
                            eval_groups[parts_idx][-1].append(g_e)

        # assign sub-eval_set components to worker parts.
        for parts_idx, e_set in eval_sets.items():
            parts[parts_idx]['eval_set'] = e_set
            if eval_names:
                parts[parts_idx]['eval_names'] = [eval_names[i] for i in range(len(e_set))]
            if eval_sample_weight:
                parts[parts_idx]['eval_sample_weight'] = eval_sample_weights[parts_idx]
            if eval_group:
                parts[parts_idx]['eval_group'] = eval_groups[parts_idx]

    # Start computation in the background
    parts = list(map(delayed, parts))
    parts = client.compute(parts)
    wait(parts)

    for part in parts:
        if part.status == 'error':  # type: ignore
            return part  # trigger error locally

    # Find locations of all parts and map them to particular Dask workers
    key_to_part_dict = {part.key: part for part in parts}  # type: ignore
    who_has = client.who_has(parts)
    worker_map = defaultdict(list)
    for key, workers in who_has.items():
        worker_map[next(iter(workers))].append(key_to_part_dict[key])

    master_worker = next(iter(worker_map))
    worker_ncores = client.ncores()

    # resolve aliases for network parameters and pop the result off params.
    # these values are added back in calls to `_train_part()`
    params = _choose_param_value(
        main_param_name="local_listen_port",
        params=params,
        default_value=12400
    )
    local_listen_port = params.pop("local_listen_port")

    params = _choose_param_value(
        main_param_name="machines",
        params=params,
        default_value=None
    )
    machines = params.pop("machines")

    # figure out network params
    worker_addresses = worker_map.keys()
    if machines is not None:
        _log_info("Using passed-in 'machines' parameter")
        worker_address_to_port = _machines_to_worker_map(
            machines=machines,
            worker_addresses=worker_addresses
        )
    else:
        if listen_port_in_params:
            _log_info("Using passed-in 'local_listen_port' for all workers")
            unique_hosts = set(urlparse(a).hostname for a in worker_addresses)
            if len(unique_hosts) < len(worker_addresses):
                msg = (
                    "'local_listen_port' was provided in Dask training parameters, but at least one "
                    "machine in the cluster has multiple Dask worker processes running on it. Please omit "
                    "'local_listen_port' or pass 'machines'."
                )
                raise LightGBMError(msg)

            worker_address_to_port = {
                address: local_listen_port
                for address in worker_addresses
            }
        else:
            _log_info("Finding random open ports for workers")
            worker_address_to_port = _find_ports_for_workers(
                client=client,
                worker_addresses=worker_addresses,
                local_listen_port=local_listen_port
            )
        machines = ','.join([
            '%s:%d' % (urlparse(worker_address).hostname, port)
            for worker_address, port
            in worker_address_to_port.items()
        ])

    num_machines = len(worker_address_to_port)

    # Tell each worker to train on the parts that it has locally
    futures_classifiers = [
        client.submit(
            _train_part,
            model_factory=model_factory,
            params={**params, 'num_threads': worker_ncores[worker]},
            list_of_parts=list_of_parts,
            machines=machines,
            local_listen_port=worker_address_to_port[worker],
            num_machines=num_machines,
            time_out=params.get('time_out', 120),
            return_model=(worker == master_worker),
            evals_provided=eval_set is not None,
            **kwargs
        )
        for worker, list_of_parts in worker_map.items()
    ]

    results = client.gather(futures_classifiers)
    results = [v for v in results if v]
    model = results[0]

    # if network parameters were changed during training, remove them from the
    # returned moodel so that they're generated dynamically on every run based
    # on the Dask cluster you're connected to and which workers have pieces of
    # the training data
    if not listen_port_in_params:
        for param in _ConfigAliases.get('local_listen_port'):
            model._other_params.pop(param, None)

    if not machines_in_params:
        for param in _ConfigAliases.get('machines'):
            model._other_params.pop(param, None)

    for param in _ConfigAliases.get('num_machines', 'timeout'):
        model._other_params.pop(param, None)

    return model


def _predict_part(
    part: _DaskPart,
    model: LGBMModel,
    raw_score: bool,
    pred_proba: bool,
    pred_leaf: bool,
    pred_contrib: bool,
    **kwargs: Any
) -> _DaskPart:

    if part.shape[0] == 0:
        result = np.array([])
    elif pred_proba:
        result = model.predict_proba(
            part,
            raw_score=raw_score,
            pred_leaf=pred_leaf,
            pred_contrib=pred_contrib,
            **kwargs
        )
    else:
        result = model.predict(
            part,
            raw_score=raw_score,
            pred_leaf=pred_leaf,
            pred_contrib=pred_contrib,
            **kwargs
        )

    # dask.DataFrame.map_partitions() expects each call to return a pandas DataFrame or Series
    if isinstance(part, pd_DataFrame):
        if pred_proba or pred_contrib or pred_leaf:
            result = pd_DataFrame(result, index=part.index)
        else:
            result = pd_Series(result, index=part.index, name='predictions')

    return result


def _predict(
    model: LGBMModel,
    data: _DaskMatrixLike,
    raw_score: bool = False,
    pred_proba: bool = False,
    pred_leaf: bool = False,
    pred_contrib: bool = False,
    dtype: _PredictionDtype = np.float32,
    **kwargs: Any
) -> dask_Array:
    """Inner predict routine.

    Parameters
    ----------
    model : lightgbm.LGBMClassifier, lightgbm.LGBMRegressor, or lightgbm.LGBMRanker class
        Fitted underlying model.
    data : Dask Array or Dask DataFrame of shape = [n_samples, n_features]
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
    predicted_result : Dask Array of shape = [n_samples] or shape = [n_samples, n_classes]
        The predicted values.
    X_leaves : Dask Array of shape = [n_samples, n_trees] or shape = [n_samples, n_trees * n_classes]
        If ``pred_leaf=True``, the predicted leaf of every tree for each sample.
    X_SHAP_values : Dask Array of shape = [n_samples, n_features + 1] or shape = [n_samples, (n_features + 1) * n_classes]
        If ``pred_contrib=True``, the feature contributions for each sample.
    """
    if not all((DASK_INSTALLED, PANDAS_INSTALLED, SKLEARN_INSTALLED)):
        raise LightGBMError('dask, pandas and scikit-learn are required for lightgbm.dask')
    if isinstance(data, dask_DataFrame):
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
        raise TypeError('Data must be either Dask Array or Dask DataFrame. Got %s.' % str(type(data)))


class _DaskLGBMModel:

    @property
    def client_(self) -> Client:
        """:obj:`dask.distributed.Client`: Dask client.

        This property can be passed in the constructor or updated
        with ``model.set_params(client=client)``.
        """
        if not getattr(self, "fitted_", False):
            raise LGBMNotFittedError('Cannot access property client_ before calling fit().')

        return _get_dask_client(client=self.client)

    def _lgb_dask_getstate(self) -> Dict[Any, Any]:
        """Remove un-picklable attributes before serialization."""
        client = self.__dict__.pop("client", None)
        self._other_params.pop("client", None)
        out = deepcopy(self.__dict__)
        out.update({"client": None})
        self.client = client
        return out

    def _lgb_dask_fit(
        self,
        model_factory: Type[LGBMModel],
        X: _DaskMatrixLike,
        y: _DaskCollection,
        sample_weight: Optional[_DaskCollection] = None,
        init_score: Optional[List[_DaskCollection]] = None,
        group: Optional[_DaskCollection] = None,
        eval_set: Optional[List[Tuple[_DaskCollection, _DaskCollection]]] = None,
        eval_names: Optional[List[str]] = None,
        eval_sample_weight: Optional[List[_DaskCollection]] = None,
        eval_class_weight: Optional[Union[dict, str]] = None,
        eval_init_score: Optional[List[_DaskCollection]] = None,
        eval_group: Optional[List[_DaskCollection]] = None,
        eval_metric: Optional[Union[Callable, str, List[Union[Callable, str]]]] = None,
        eval_stopping_rounds: Optional[int] = None,
        **kwargs: Any
    ) -> "_DaskLGBMModel":
        if not all((DASK_INSTALLED, PANDAS_INSTALLED, SKLEARN_INSTALLED)):
            raise LightGBMError('dask, pandas and scikit-learn are required for lightgbm.dask')

        not_supported = ['init_score', 'eval_init_score', 'eval_class_weight']
        for ns in not_supported:
            if eval(ns) is not None:
                raise RuntimeError(f'{ns} is not currently supported in lightgbm.dask')

        params = self.get_params(True)
        params.pop("client", None)

        # easier to pass fit args as kwargs than to list_of_parts in _train.
        if eval_metric:
            kwargs['eval_metric'] = eval_metric
        if eval_stopping_rounds:
            kwargs['eval_stopping_rounds'] = eval_stopping_rounds

        model = _train(
            client=_get_dask_client(self.client),
            data=X,
            label=y,
            params=params,
            model_factory=model_factory,
            sample_weight=sample_weight,
            group=group,
            eval_set=eval_set,
            eval_names=eval_names,
            eval_sample_weight=eval_sample_weight,
            eval_group=eval_group,
            **kwargs
        )

        self.set_params(**model.get_params())
        self._lgb_dask_copy_extra_params(model, self)

        return self

    def _lgb_dask_to_local(self, model_factory: Type[LGBMModel]) -> LGBMModel:
        params = self.get_params()
        params.pop("client", None)
        model = model_factory(**params)
        self._lgb_dask_copy_extra_params(self, model)
        model._other_params.pop("client", None)
        return model

    @staticmethod
    def _lgb_dask_copy_extra_params(source: Union["_DaskLGBMModel", LGBMModel], dest: Union["_DaskLGBMModel", LGBMModel]) -> None:
        params = source.get_params()
        attributes = source.__dict__
        extra_param_names = set(attributes.keys()).difference(params.keys())
        for name in extra_param_names:
            setattr(dest, name, attributes[name])


class DaskLGBMClassifier(LGBMClassifier, _DaskLGBMModel):
    """Distributed version of lightgbm.LGBMClassifier."""

    def __init__(
        self,
        boosting_type: str = 'gbdt',
        num_leaves: int = 31,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample_for_bin: int = 200000,
        objective: Optional[Union[Callable, str]] = None,
        class_weight: Optional[Union[dict, str]] = None,
        min_split_gain: float = 0.,
        min_child_weight: float = 1e-3,
        min_child_samples: int = 20,
        subsample: float = 1.,
        subsample_freq: int = 0,
        colsample_bytree: float = 1.,
        reg_alpha: float = 0.,
        reg_lambda: float = 0.,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        n_jobs: int = -1,
        silent: bool = True,
        importance_type: str = 'split',
        client: Optional[Client] = None,
        **kwargs: Any
    ):
        """Docstring is inherited from the lightgbm.LGBMClassifier.__init__."""
        self.client = client
        super().__init__(
            boosting_type=boosting_type,
            num_leaves=num_leaves,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample_for_bin=subsample_for_bin,
            objective=objective,
            class_weight=class_weight,
            min_split_gain=min_split_gain,
            min_child_weight=min_child_weight,
            min_child_samples=min_child_samples,
            subsample=subsample,
            subsample_freq=subsample_freq,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            n_jobs=n_jobs,
            silent=silent,
            importance_type=importance_type,
            **kwargs
        )

    _base_doc = LGBMClassifier.__init__.__doc__
    _before_kwargs, _kwargs, _after_kwargs = _base_doc.partition('**kwargs')
    _base_doc = (
        _before_kwargs
        + 'client : dask.distributed.Client or None, optional (default=None)\n'
        + ' ' * 12 + 'Dask client. If ``None``, ``distributed.default_client()`` will be used at runtime. The Dask client used by this class will not be saved if the model object is pickled.\n'
        + ' ' * 8 + _kwargs + _after_kwargs
    )

    # the note on custom objective functions in LGBMModel.__init__ is not
    # currently relevant for the Dask estimators
    __init__.__doc__ = _base_doc[:_base_doc.find('Note\n')]

    def __getstate__(self) -> Dict[Any, Any]:
        return self._lgb_dask_getstate()

    def fit(
        self,
        X: _DaskMatrixLike,
        y: _DaskCollection,
        sample_weight: Optional[_DaskCollection] = None,
        init_score: Optional[_DaskCollection] = None,
        eval_set: Optional[List[Tuple[_DaskCollection, _DaskCollection]]] = None,
        eval_names: Optional[List[str]] = None,
        eval_sample_weight: Optional[List[_DaskCollection]] = None,
        eval_class_weight: Optional[List[_DaskCollection]] = None,
        eval_init_score: Optional[List[_DaskCollection]] = None,
        eval_metric: Optional[Union[Callable, str, List[Union[Callable, str]]]] = None,
        eval_stopping_rounds: Optional[int] = None,
        **kwargs: Any
    ) -> "DaskLGBMClassifier":
        """Docstring is inherited from the lightgbm.LGBMClassifier.fit."""
        not_supported = ['init_score', 'eval_class_weight', 'eval_init_score']
        for ns in not_supported:
            if eval(ns) is not None:
                raise RuntimeError(f'{ns} is not currently supported in lightgbm.dask')

        if eval_metric:
            kwargs['eval_metric'] = eval_metric
        if eval_stopping_rounds:
            kwargs['eval_stopping_rounds'] = eval_stopping_rounds

        return self._lgb_dask_fit(
            model_factory=LGBMClassifier,
            X=X,
            y=y,
            sample_weight=sample_weight,
            eval_set=eval_set,
            eval_names=eval_names,
            eval_sample_weight=eval_sample_weight,
            **kwargs
        )

    _base_doc = _lgbmmodel_doc_fit.format(
        X_shape="Dask Array or Dask DataFrame of shape = [n_samples, n_features]",
        y_shape="Dask Array, Dask DataFrame or Dask Series of shape = [n_samples]",
        sample_weight_shape="Dask Array, Dask DataFrame, Dask Series of shape = [n_samples] or None, optional (default=None)",
        group_shape="Dask Array, Dask DataFrame, Dask Series of shape = [n_samples] or None, optional (default=None)",
        eval_sample_weight_shape='List of Dask Arrays, Dask DataFrames, Dask Series or None, optional (default=None)',
        eval_init_score_shape='List of Dask Arrays, Dask DataFrames, Dask Series or None, optional (default=None)',
        eval_group_shape='List of Dask Arrays, Dask DataFrames, Dask Series or None, optional (default=None)'
    )

    # DaskLGBMClassifier does not support init_score, eval_class_weight, or eval_init_score
    _base_doc = (_base_doc[:_base_doc.find('init_score :')]
                 + _base_doc[_base_doc.find('init_score :'):])

    _base_doc = (_base_doc[:_base_doc.find('eval_class_weight :')]
                 + _base_doc[_base_doc.find('eval_init_score :'):])

    # DaskLGBMClassifier support for callbacks and init_model is not tested
    fit.__doc__ = (
        _base_doc[:_base_doc.find('callbacks :')]
        + '**kwargs\n'
        + ' ' * 12 + 'Other parameters passed through to ``LGBMClassifier.fit()``.\n'
    )

    def predict(self, X: _DaskMatrixLike, **kwargs: Any) -> dask_Array:
        """Docstring is inherited from the lightgbm.LGBMClassifier.predict."""
        return _predict(
            model=self.to_local(),
            data=X,
            dtype=self.classes_.dtype,
            **kwargs
        )

    predict.__doc__ = _lgbmmodel_doc_predict.format(
        description="Return the predicted value for each sample.",
        X_shape="Dask Array or Dask DataFrame of shape = [n_samples, n_features]",
        output_name="predicted_result",
        predicted_result_shape="Dask Array of shape = [n_samples] or shape = [n_samples, n_classes]",
        X_leaves_shape="Dask Array of shape = [n_samples, n_trees] or shape = [n_samples, n_trees * n_classes]",
        X_SHAP_values_shape="Dask Array of shape = [n_samples, n_features + 1] or shape = [n_samples, (n_features + 1) * n_classes]"
    )

    def predict_proba(self, X: _DaskMatrixLike, **kwargs: Any) -> dask_Array:
        """Docstring is inherited from the lightgbm.LGBMClassifier.predict_proba."""
        return _predict(
            model=self.to_local(),
            data=X,
            pred_proba=True,
            **kwargs
        )

    predict_proba.__doc__ = _lgbmmodel_doc_predict.format(
        description="Return the predicted probability for each class for each sample.",
        X_shape="Dask Array or Dask DataFrame of shape = [n_samples, n_features]",
        output_name="predicted_probability",
        predicted_result_shape="Dask Array of shape = [n_samples] or shape = [n_samples, n_classes]",
        X_leaves_shape="Dask Array of shape = [n_samples, n_trees] or shape = [n_samples, n_trees * n_classes]",
        X_SHAP_values_shape="Dask Array of shape = [n_samples, n_features + 1] or shape = [n_samples, (n_features + 1) * n_classes]"
    )

    def to_local(self) -> LGBMClassifier:
        """Create regular version of lightgbm.LGBMClassifier from the distributed version.

        Returns
        -------
        model : lightgbm.LGBMClassifier
            Local underlying model.
        """
        return self._lgb_dask_to_local(LGBMClassifier)


class DaskLGBMRegressor(LGBMRegressor, _DaskLGBMModel):
    """Distributed version of lightgbm.LGBMRegressor."""

    def __init__(
        self,
        boosting_type: str = 'gbdt',
        num_leaves: int = 31,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample_for_bin: int = 200000,
        objective: Optional[Union[Callable, str]] = None,
        class_weight: Optional[Union[dict, str]] = None,
        min_split_gain: float = 0.,
        min_child_weight: float = 1e-3,
        min_child_samples: int = 20,
        subsample: float = 1.,
        subsample_freq: int = 0,
        colsample_bytree: float = 1.,
        reg_alpha: float = 0.,
        reg_lambda: float = 0.,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        n_jobs: int = -1,
        silent: bool = True,
        importance_type: str = 'split',
        client: Optional[Client] = None,
        **kwargs: Any
    ):
        """Docstring is inherited from the lightgbm.LGBMRegressor.__init__."""
        self.client = client
        super().__init__(
            boosting_type=boosting_type,
            num_leaves=num_leaves,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample_for_bin=subsample_for_bin,
            objective=objective,
            class_weight=class_weight,
            min_split_gain=min_split_gain,
            min_child_weight=min_child_weight,
            min_child_samples=min_child_samples,
            subsample=subsample,
            subsample_freq=subsample_freq,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            n_jobs=n_jobs,
            silent=silent,
            importance_type=importance_type,
            **kwargs
        )

    _base_doc = LGBMRegressor.__init__.__doc__
    _before_kwargs, _kwargs, _after_kwargs = _base_doc.partition('**kwargs')
    _base_doc = (
        _before_kwargs
        + 'client : dask.distributed.Client or None, optional (default=None)\n'
        + ' ' * 12 + 'Dask client. If ``None``, ``distributed.default_client()`` will be used at runtime. The Dask client used by this class will not be saved if the model object is pickled.\n'
        + ' ' * 8 + _kwargs + _after_kwargs
    )

    # the note on custom objective functions in LGBMModel.__init__ is not
    # currently relevant for the Dask estimators
    __init__.__doc__ = _base_doc[:_base_doc.find('Note\n')]

    def __getstate__(self) -> Dict[Any, Any]:
        return self._lgb_dask_getstate()

    def fit(
        self,
        X: _DaskMatrixLike,
        y: _DaskCollection,
        sample_weight: Optional[_DaskCollection] = None,
        init_score: Optional[_DaskCollection] = None,
        eval_set: Optional[List[Tuple[_DaskCollection, _DaskCollection]]] = None,
        eval_names: Optional[List[str]] = None,
        eval_sample_weight: Optional[List[_DaskCollection]] = None,
        eval_init_score: Optional[List[_DaskCollection]] = None,
        eval_metric: Optional[Union[Callable, str, List[Union[Callable, str]]]] = None,
        eval_stopping_rounds: Optional[int] = None,
        **kwargs: Any
    ) -> "DaskLGBMRegressor":
        """Docstring is inherited from the lightgbm.LGBMRegressor.fit."""
        not_supported = ['init_score', 'eval_init_score']
        for ns in not_supported:
            if eval(ns) is not None:
                raise RuntimeError(f'{ns} is not currently supported in lightgbm.dask')

        if eval_metric:
            kwargs['eval_metric'] = eval_metric
        if eval_stopping_rounds:
            kwargs['eval_stopping_rounds'] = eval_stopping_rounds

        return self._lgb_dask_fit(
            model_factory=LGBMRegressor,
            X=X,
            y=y,
            sample_weight=sample_weight,
            eval_set=eval_set,
            eval_names=eval_names,
            eval_sample_weight=eval_sample_weight,
            **kwargs
        )

    _base_doc = _lgbmmodel_doc_fit.format(
        X_shape="Dask Array or Dask DataFrame of shape = [n_samples, n_features]",
        y_shape="Dask Array, Dask DataFrame or Dask Series of shape = [n_samples]",
        sample_weight_shape="Dask Array, Dask DataFrame, Dask Series of shape = [n_samples] or None, optional (default=None)",
        group_shape="Dask Array, Dask DataFrame, Dask Series of shape = [n_samples] or None, optional (default=None)",
        eval_sample_weight_shape='List of Dask Arrays, Dask DataFrames, Dask Series or None, optional (default=None)',
        eval_init_score_shape='List of Dask Arrays, Dask DataFrames, Dask Series or None, optional (default=None)',
        eval_group_shape='List of Dask Arrays, Dask DataFrames, Dask Series or None, optional (default=None)'
    )

    # DaskLGBMRegressor does not support init_score or eval_init_score
    _base_doc = (_base_doc[:_base_doc.find('init_score :')]
                 + _base_doc[_base_doc.find('init_score :'):])

    _base_doc = (_base_doc[:_base_doc.find('eval_init_weight :')]
                 + _base_doc[_base_doc.find('eval_init_score :'):])

    # DaskLGBMRegressor support for callbacks and init_model is not tested
    fit.__doc__ = (
        _base_doc[:_base_doc.find('callbacks :')]
        + '**kwargs\n'
        + ' ' * 12 + 'Other parameters passed through to ``LGBMRegressor.fit()``.\n'
    )

    def predict(self, X: _DaskMatrixLike, **kwargs) -> dask_Array:
        """Docstring is inherited from the lightgbm.LGBMRegressor.predict."""
        return _predict(
            model=self.to_local(),
            data=X,
            **kwargs
        )

    predict.__doc__ = _lgbmmodel_doc_predict.format(
        description="Return the predicted value for each sample.",
        X_shape="Dask Array or Dask DataFrame of shape = [n_samples, n_features]",
        output_name="predicted_result",
        predicted_result_shape="Dask Array of shape = [n_samples]",
        X_leaves_shape="Dask Array of shape = [n_samples, n_trees]",
        X_SHAP_values_shape="Dask Array of shape = [n_samples, n_features + 1]"
    )

    def to_local(self) -> LGBMRegressor:
        """Create regular version of lightgbm.LGBMRegressor from the distributed version.

        Returns
        -------
        model : lightgbm.LGBMRegressor
            Local underlying model.
        """
        return self._lgb_dask_to_local(LGBMRegressor)


class DaskLGBMRanker(LGBMRanker, _DaskLGBMModel):
    """Distributed version of lightgbm.LGBMRanker."""

    def __init__(
        self,
        boosting_type: str = 'gbdt',
        num_leaves: int = 31,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample_for_bin: int = 200000,
        objective: Optional[Union[Callable, str]] = None,
        class_weight: Optional[Union[dict, str]] = None,
        min_split_gain: float = 0.,
        min_child_weight: float = 1e-3,
        min_child_samples: int = 20,
        subsample: float = 1.,
        subsample_freq: int = 0,
        colsample_bytree: float = 1.,
        reg_alpha: float = 0.,
        reg_lambda: float = 0.,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        n_jobs: int = -1,
        silent: bool = True,
        importance_type: str = 'split',
        client: Optional[Client] = None,
        **kwargs: Any
    ):
        """Docstring is inherited from the lightgbm.LGBMRanker.__init__."""
        self.client = client
        super().__init__(
            boosting_type=boosting_type,
            num_leaves=num_leaves,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample_for_bin=subsample_for_bin,
            objective=objective,
            class_weight=class_weight,
            min_split_gain=min_split_gain,
            min_child_weight=min_child_weight,
            min_child_samples=min_child_samples,
            subsample=subsample,
            subsample_freq=subsample_freq,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            n_jobs=n_jobs,
            silent=silent,
            importance_type=importance_type,
            **kwargs
        )

    _base_doc = LGBMRanker.__init__.__doc__
    _before_kwargs, _kwargs, _after_kwargs = _base_doc.partition('**kwargs')
    _base_doc = (
        _before_kwargs
        + 'client : dask.distributed.Client or None, optional (default=None)\n'
        + ' ' * 12 + 'Dask client. If ``None``, ``distributed.default_client()`` will be used at runtime. The Dask client used by this class will not be saved if the model object is pickled.\n'
        + ' ' * 8 + _kwargs + _after_kwargs
    )

    # the note on custom objective functions in LGBMModel.__init__ is not
    # currently relevant for the Dask estimators
    __init__.__doc__ = _base_doc[:_base_doc.find('Note\n')]

    def __getstate__(self) -> Dict[Any, Any]:
        return self._lgb_dask_getstate()

    def fit(
        self,
        X: _DaskMatrixLike,
        y: _DaskCollection,
        sample_weight: Optional[_DaskCollection] = None,
        init_score: Optional[_DaskCollection] = None,
        group: Optional[_DaskCollection] = None,
        eval_set: Optional[List[Tuple[_DaskCollection, _DaskCollection]]] = None,
        eval_names: Optional[List[str]] = None,
        eval_sample_weight: Optional[List[_DaskCollection]] = None,
        eval_class_weight: Optional[List[_DaskCollection]] = None,
        eval_init_score: Optional[List[_DaskCollection]] = None,
        eval_group: Optional[List[_DaskCollection]] = None,
        eval_metric: Optional[Union[Callable, str, List[Union[Callable, str]]]] = None,
        eval_at: Optional[List[int]] = None,
        eval_stopping_rounds: Optional[int] = None,
        **kwargs: Any
    ) -> "DaskLGBMRanker":
        """Docstring is inherited from the lightgbm.LGBMRanker.fit."""
        not_supported = ['init_score', 'eval_init_score', 'eval_class_weight']
        for ns in not_supported:
            if eval(ns) is not None:
                raise RuntimeError(f'{ns} is not currently supported in lightgbm.dask')

        if eval_metric:
            kwargs['eval_metric'] = eval_metric
        if eval_at:
            kwargs['eval_at'] = eval_at
        if eval_stopping_rounds:
            kwargs['eval_stopping_rounds'] = eval_stopping_rounds

        return self._lgb_dask_fit(
            model_factory=LGBMRanker,
            X=X,
            y=y,
            sample_weight=sample_weight,
            group=group,
            eval_set=eval_set,
            eval_names=eval_names,
            eval_sample_weight=eval_sample_weight,
            eval_group=eval_group,
            **kwargs
        )

    _base_doc = _lgbmmodel_doc_fit.format(
        X_shape="Dask Array or Dask DataFrame of shape = [n_samples, n_features]",
        y_shape="Dask Array, Dask DataFrame or Dask Series of shape = [n_samples]",
        sample_weight_shape="Dask Array, Dask DataFrame, Dask Series of shape = [n_samples] or None, optional (default=None)",
        group_shape="Dask Array, Dask DataFrame, Dask Series of shape = [n_samples] or None, optional (default=None)",
        eval_sample_weight_shape='List of Dask Arrays, Dask DataFrames, Dask Series or None, optional (default=None)',
        eval_init_score_shape='List of Dask Arrays, Dask DataFrames, Dask Series or None, optional (default=None)',
        eval_group_shape='List of Dask Arrays, Dask DataFrames, Dask Series or None, optional (default=None)'
    )

    # DaskLGBMRanker does not support init_score or eval_init_score.
    _base_doc = (_base_doc[:_base_doc.find('init_score :')]
                 + _base_doc[_base_doc.find('init_score :'):])

    _base_doc = (_base_doc[:_base_doc.find('eval_init_score :')]
                 + _base_doc[_base_doc.find('eval_init_score :'):])

    # DaskLGBMRanker support for callbacks and init_model is not tested
    fit.__doc__ = (
        _base_doc[:_base_doc.find('callbacks :')]
        + '**kwargs\n'
        + ' ' * 12 + 'Other parameters passed through to ``LGBMRanker.fit()``.\n'
    )

    def predict(self, X: _DaskMatrixLike, **kwargs: Any) -> dask_Array:
        """Docstring is inherited from the lightgbm.LGBMRanker.predict."""
        return _predict(self.to_local(), X, **kwargs)

    predict.__doc__ = _lgbmmodel_doc_predict.format(
        description="Return the predicted value for each sample.",
        X_shape="Dask Array or Dask DataFrame of shape = [n_samples, n_features]",
        output_name="predicted_result",
        predicted_result_shape="Dask Array of shape = [n_samples]",
        X_leaves_shape="Dask Array of shape = [n_samples, n_trees]",
        X_SHAP_values_shape="Dask Array of shape = [n_samples, n_features + 1]"
    )

    def to_local(self) -> LGBMRanker:
        """Create regular version of lightgbm.LGBMRanker from the distributed version.

        Returns
        -------
        model : lightgbm.LGBMRanker
            Local underlying model.
        """
        return self._lgb_dask_to_local(LGBMRanker)
