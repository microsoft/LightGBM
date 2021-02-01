# coding: utf-8
"""Distributed training with LightGBM and Dask.distributed.

This module enables you to perform distributed training with LightGBM on
Dask.Array and Dask.DataFrame collections.

It is based on dask-lightgbm, which was based on dask-xgboost.
"""
import socket
from collections import defaultdict
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List, Optional, Type, Union
from urllib.parse import urlparse

import numpy as np
import scipy.sparse as ss

from .basic import _choose_param_value, _ConfigAliases, _LIB, _log_warning, _safe_call, LightGBMError
from .compat import (PANDAS_INSTALLED, pd_DataFrame, pd_Series, concat,
                     SKLEARN_INSTALLED,
                     DASK_INSTALLED, dask_DataFrame, dask_Array, dask_Series, delayed, Client, default_client, get_worker, wait)
from .sklearn import LGBMClassifier, LGBMModel, LGBMRegressor, LGBMRanker

_DaskCollection = Union[dask_Array, dask_DataFrame, dask_Series]
_DaskMatrixLike = Union[dask_Array, dask_DataFrame]
_DaskPart = Union[np.ndarray, pd_DataFrame, pd_Series, ss.spmatrix]
_PredictionDtype = Union[Type[np.float32], Type[np.float64], Type[np.int32], Type[np.int64]]


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
    worker_address_to_port: Dict[str, int],
    return_model: bool,
    time_out: int = 120,
    **kwargs: Any
) -> Optional[LGBMModel]:
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


def _split_to_parts(data: _DaskCollection, is_matrix: bool) -> List[_DaskPart]:
    parts = data.to_delayed()
    if isinstance(parts, np.ndarray):
        if is_matrix:
            assert parts.shape[1] == 1
        else:
            assert parts.ndim == 1 or parts.shape[1] == 1
        parts = parts.flatten().tolist()
    return parts


def _train(
    client: Client,
    data: _DaskMatrixLike,
    label: _DaskCollection,
    params: Dict[str, Any],
    model_factory: Type[LGBMModel],
    sample_weight: Optional[_DaskCollection] = None,
    group: Optional[_DaskCollection] = None,
    **kwargs: Any
) -> LGBMModel:
    """Inner train routine.

    Parameters
    ----------
    client : dask.distributed.Client
        Dask client.
    data : dask Array or dask DataFrame of shape = [n_samples, n_features]
        Input feature matrix.
    label : dask Array, dask DataFrame or dask Series of shape = [n_samples]
        The target values (class labels in classification, real numbers in regression).
    params : dict
        Parameters passed to constructor of the local underlying model.
    model_factory : lightgbm.LGBMClassifier, lightgbm.LGBMRegressor, or lightgbm.LGBMRanker class
        Class of the local underlying model.
    sample_weight : dask Array, dask DataFrame, Dask Series of shape = [n_samples] or None, optional (default=None)
        Weights of training data.
    group : dask Array, dask DataFrame, Dask Series of shape = [n_samples] or None, optional (default=None)
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
        _log_warning('Parameter tree_learner set to %s, which is not allowed. Using "data" as default' % params['tree_learner'])
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


def _predict_part(
    part: _DaskPart,
    model: LGBMModel,
    raw_score: bool,
    pred_proba: bool,
    pred_leaf: bool,
    pred_contrib: bool,
    **kwargs: Any
) -> _DaskPart:
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
    data : dask Array or dask DataFrame of shape = [n_samples, n_features]
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
    predicted_result : dask Array of shape = [n_samples] or shape = [n_samples, n_classes]
        The predicted values.
    X_leaves : dask Array of shape = [n_samples, n_trees] or shape = [n_samples, n_trees * n_classes]
        If ``pred_leaf=True``, the predicted leaf of every tree for each sample.
    X_SHAP_values : dask Array of shape = [n_samples, n_features + 1] or shape = [n_samples, (n_features + 1) * n_classes]
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
        raise TypeError('Data must be either dask Array or dask DataFrame. Got %s.' % str(type(data)))


class _DaskLGBMModel:

    # self._client is set in the constructor of classes that use this mixin
    _client: Optional[Client] = None

    @property
    def client_(self) -> Client:
        """Dask client.

        This property can be passed in the constructor or updated
        with ``model.set_params(client=client)``.
        """
        if self._client is None:
            return default_client()
        else:
            return self._client

    def _lgb_getstate(self) -> Dict[Any, Any]:
        """Remove un-picklable attributes before serialization."""
        client = self.__dict__.pop("client", None)
        self.__dict__.pop("_client", None)
        self._other_params.pop("client", None)
        out = deepcopy(self.__dict__)
        out.update({"_client": None, "client": None})
        self._client = client
        self.client = client
        return out

    def _fit(
        self,
        model_factory: Type[LGBMModel],
        X: _DaskMatrixLike,
        y: _DaskCollection,
        sample_weight: Optional[_DaskCollection] = None,
        group: Optional[_DaskCollection] = None,
        **kwargs: Any
    ) -> "_DaskLGBMModel":
        if not all((DASK_INSTALLED, PANDAS_INSTALLED, SKLEARN_INSTALLED)):
            raise LightGBMError('dask, pandas and scikit-learn are required for lightgbm.dask')

        params = self.get_params(True)
        params.pop("client", None)

        model = _train(
            client=self.client_,
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

    def _to_local(self, model_factory: Type[LGBMModel]) -> LGBMModel:
        params = self.get_params()
        params.pop("client", None)
        model = model_factory(**params)
        self._copy_extra_params(self, model)
        model._other_params.pop("client", None)
        return model

    @staticmethod
    def _copy_extra_params(source: Union["_DaskLGBMModel", LGBMModel], dest: Union["_DaskLGBMModel", LGBMModel]) -> None:
        params = source.get_params()
        attributes = source.__dict__
        extra_param_names = set(attributes.keys()).difference(params.keys())
        for name in extra_param_names:
            if name != "_client":
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
        self._client = client
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
    __init__.__doc__ = (
        _before_kwargs
        + 'client : dask.distributed.Client or None, optional (default=None)\n'
        + ' ' * 12 + 'Dask client. If ``None``, ``distributed.default_client()`` will be used at runtime. This client will not be saved if the model object is pickled.\n'
        + ' ' * 8 + _kwargs + _after_kwargs
    )

    def __getstate__(self) -> Dict[Any, Any]:
        return self._lgb_getstate()

    def fit(
        self,
        X: _DaskMatrixLike,
        y: _DaskCollection,
        sample_weight: Optional[_DaskCollection] = None,
        **kwargs: Any
    ) -> "DaskLGBMClassifier":
        """Docstring is inherited from the lightgbm.LGBMClassifier.fit."""
        return self._fit(
            model_factory=LGBMClassifier,
            X=X,
            y=y,
            sample_weight=sample_weight,
            **kwargs
        )

    fit.__doc__ = LGBMClassifier.fit.__doc__

    def predict(self, X: _DaskMatrixLike, **kwargs: Any) -> dask_Array:
        """Docstring is inherited from the lightgbm.LGBMClassifier.predict."""
        return _predict(
            model=self.to_local(),
            data=X,
            dtype=self.classes_.dtype,
            **kwargs
        )

    predict.__doc__ = LGBMClassifier.predict.__doc__

    def predict_proba(self, X: _DaskMatrixLike, **kwargs: Any) -> dask_Array:
        """Docstring is inherited from the lightgbm.LGBMClassifier.predict_proba."""
        return _predict(
            model=self.to_local(),
            data=X,
            pred_proba=True,
            **kwargs
        )

    predict_proba.__doc__ = LGBMClassifier.predict_proba.__doc__

    def to_local(self) -> LGBMClassifier:
        """Create regular version of lightgbm.LGBMClassifier from the distributed version.

        Returns
        -------
        model : lightgbm.LGBMClassifier
            Local underlying model.
        """
        return self._to_local(LGBMClassifier)


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
        self._client = client
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
    __init__.__doc__ = (
        _before_kwargs
        + 'client : dask.distributed.Client or None, optional (default=None)\n'
        + ' ' * 12 + 'Dask client. If ``None``, ``distributed.default_client()`` will be used at runtime. This client will not be saved if the model object is pickled.\n'
        + ' ' * 8 + _kwargs + _after_kwargs
    )

    def __getstate__(self) -> Dict[Any, Any]:
        return self._lgb_getstate()

    def fit(
        self,
        X: _DaskMatrixLike,
        y: _DaskCollection,
        sample_weight: Optional[_DaskCollection] = None,
        client: Optional[Client] = None,
        **kwargs: Any
    ) -> "DaskLGBMRegressor":
        """Docstring is inherited from the lightgbm.LGBMRegressor.fit."""
        return self._fit(
            model_factory=LGBMRegressor,
            X=X,
            y=y,
            sample_weight=sample_weight,
            **kwargs
        )

    fit.__doc__ = LGBMRegressor.fit.__doc__

    def predict(self, X: _DaskMatrixLike, **kwargs) -> dask_Array:
        """Docstring is inherited from the lightgbm.LGBMRegressor.predict."""
        return _predict(
            model=self.to_local(),
            data=X,
            **kwargs
        )

    predict.__doc__ = LGBMRegressor.predict.__doc__

    def to_local(self) -> LGBMRegressor:
        """Create regular version of lightgbm.LGBMRegressor from the distributed version.

        Returns
        -------
        model : lightgbm.LGBMRegressor
            Local underlying model.
        """
        return self._to_local(LGBMRegressor)


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
        self._client = client
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
    __init__.__doc__ = (
        _before_kwargs
        + 'client : dask.distributed.Client or None, optional (default=None)\n'
        + ' ' * 12 + 'Dask client. If ``None``, ``distributed.default_client()`` will be used at runtime. This client will not be saved if the model object is pickled.\n'
        + ' ' * 8 + _kwargs + _after_kwargs
    )

    def __getstate__(self) -> Dict[Any, Any]:
        return self._lgb_getstate()

    def fit(
        self,
        X: _DaskMatrixLike,
        y: _DaskCollection,
        sample_weight: Optional[_DaskCollection] = None,
        init_score: Optional[_DaskCollection] = None,
        group: Optional[_DaskCollection] = None,
        **kwargs: Any
    ) -> "DaskLGBMRanker":
        """Docstring is inherited from the lightgbm.LGBMRanker.fit."""
        if init_score is not None:
            raise RuntimeError('init_score is not currently supported in lightgbm.dask')

        return self._fit(
            model_factory=LGBMRanker,
            X=X,
            y=y,
            sample_weight=sample_weight,
            group=group,
            **kwargs
        )

    fit.__doc__ = LGBMRanker.fit.__doc__

    def predict(self, X: _DaskMatrixLike, **kwargs: Any) -> dask_Array:
        """Docstring is inherited from the lightgbm.LGBMRanker.predict."""
        return _predict(self.to_local(), X, **kwargs)

    predict.__doc__ = LGBMRanker.predict.__doc__

    def to_local(self) -> LGBMRanker:
        """Create regular version of lightgbm.LGBMRanker from the distributed version.

        Returns
        -------
        model : lightgbm.LGBMRanker
            Local underlying model.
        """
        return self._to_local(LGBMRanker)
