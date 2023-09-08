# coding: utf-8
"""Library with training routines of LightGBM."""
import copy
import json
from collections import OrderedDict, defaultdict
from operator import attrgetter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from . import callback
from .basic import (Booster, Dataset, LightGBMError, _choose_param_value, _ConfigAliases, _InnerPredictor,
                    _LGBM_BoosterEvalMethodResultType, _LGBM_CategoricalFeatureConfiguration,
                    _LGBM_CustomObjectiveFunction, _LGBM_EvalFunctionResultType, _LGBM_FeatureNameConfiguration,
                    _log_warning)
from .compat import SKLEARN_INSTALLED, _LGBMBaseCrossValidator, _LGBMGroupKFold, _LGBMStratifiedKFold

__all__ = [
    'cv',
    'CVBooster',
    'train',
]


_LGBM_CustomMetricFunction = Union[
    Callable[
        [np.ndarray, Dataset],
        _LGBM_EvalFunctionResultType,
    ],
    Callable[
        [np.ndarray, Dataset],
        List[_LGBM_EvalFunctionResultType]
    ],
]

_LGBM_PreprocFunction = Callable[
    [Dataset, Dataset, Dict[str, Any]],
    Tuple[Dataset, Dataset, Dict[str, Any]]
]


def train(
    params: Dict[str, Any],
    train_set: Dataset,
    num_boost_round: int = 100,
    valid_sets: Optional[List[Dataset]] = None,
    valid_names: Optional[List[str]] = None,
    feval: Optional[Union[_LGBM_CustomMetricFunction, List[_LGBM_CustomMetricFunction]]] = None,
    init_model: Optional[Union[str, Path, Booster]] = None,
    feature_name: _LGBM_FeatureNameConfiguration = 'auto',
    categorical_feature: _LGBM_CategoricalFeatureConfiguration = 'auto',
    keep_training_booster: bool = False,
    callbacks: Optional[List[Callable]] = None
) -> Booster:
    """Perform the training with given parameters.

    Parameters
    ----------
    params : dict
        Parameters for training. Values passed through ``params`` take precedence over those
        supplied via arguments.
    train_set : Dataset
        Data to be trained on.
    num_boost_round : int, optional (default=100)
        Number of boosting iterations.
    valid_sets : list of Dataset, or None, optional (default=None)
        List of data to be evaluated on during training.
    valid_names : list of str, or None, optional (default=None)
        Names of ``valid_sets``.
    feval : callable, list of callable, or None, optional (default=None)
        Customized evaluation function.
        Each evaluation function should accept two parameters: preds, eval_data,
        and return (eval_name, eval_result, is_higher_better) or list of such tuples.

            preds : numpy 1-D array or numpy 2-D array (for multi-class task)
                The predicted values.
                For multi-class task, preds are numpy 2-D array of shape = [n_samples, n_classes].
                If custom objective function is used, predicted values are returned before any transformation,
                e.g. they are raw margin instead of probability of positive class for binary task in this case.
            eval_data : Dataset
                A ``Dataset`` to evaluate.
            eval_name : str
                The name of evaluation function (without whitespaces).
            eval_result : float
                The eval result.
            is_higher_better : bool
                Is eval result higher better, e.g. AUC is ``is_higher_better``.

        To ignore the default metric corresponding to the used objective,
        set the ``metric`` parameter to the string ``"None"`` in ``params``.
    init_model : str, pathlib.Path, Booster or None, optional (default=None)
        Filename of LightGBM model or Booster instance used for continue training.
    feature_name : list of str, or 'auto', optional (default="auto")
        Feature names.
        If 'auto' and data is pandas DataFrame, data columns names are used.
    categorical_feature : list of str or int, or 'auto', optional (default="auto")
        Categorical features.
        If list of int, interpreted as indices.
        If list of str, interpreted as feature names (need to specify ``feature_name`` as well).
        If 'auto' and data is pandas DataFrame, pandas unordered categorical columns are used.
        All values in categorical features will be cast to int32 and thus should be less than int32 max value (2147483647).
        Large values could be memory consuming. Consider using consecutive integers starting from zero.
        All negative values in categorical features will be treated as missing values.
        The output cannot be monotonically constrained with respect to a categorical feature.
        Floating point numbers in categorical features will be rounded towards 0.
    keep_training_booster : bool, optional (default=False)
        Whether the returned Booster will be used to keep training.
        If False, the returned value will be converted into _InnerPredictor before returning.
        This means you won't be able to use ``eval``, ``eval_train`` or ``eval_valid`` methods of the returned Booster.
        When your model is very large and cause the memory error,
        you can try to set this param to ``True`` to avoid the model conversion performed during the internal call of ``model_to_string``.
        You can still use _InnerPredictor as ``init_model`` for future continue training.
    callbacks : list of callable, or None, optional (default=None)
        List of callback functions that are applied at each iteration.
        See Callbacks in Python API for more information.

    Note
    ----
    A custom objective function can be provided for the ``objective`` parameter.
    It should accept two parameters: preds, train_data and return (grad, hess).

        preds : numpy 1-D array or numpy 2-D array (for multi-class task)
            The predicted values.
            Predicted values are returned before any transformation,
            e.g. they are raw margin instead of probability of positive class for binary task.
        train_data : Dataset
            The training dataset.
        grad : numpy 1-D array or numpy 2-D array (for multi-class task)
            The value of the first order derivative (gradient) of the loss
            with respect to the elements of preds for each sample point.
        hess : numpy 1-D array or numpy 2-D array (for multi-class task)
            The value of the second order derivative (Hessian) of the loss
            with respect to the elements of preds for each sample point.

    For multi-class task, preds are numpy 2-D array of shape = [n_samples, n_classes],
    and grad and hess should be returned in the same format.

    Returns
    -------
    booster : Booster
        The trained Booster model.
    """
    if not isinstance(train_set, Dataset):
        raise TypeError(f"train() only accepts Dataset object, train_set has type '{type(train_set).__name__}'.")

    if num_boost_round <= 0:
        raise ValueError(f"num_boost_round must be greater than 0. Got {num_boost_round}.")

    if isinstance(valid_sets, list):
        for i, valid_item in enumerate(valid_sets):
            if not isinstance(valid_item, Dataset):
                raise TypeError(
                    "Every item in valid_sets must be a Dataset object. "
                    f"Item {i} has type '{type(valid_item).__name__}'."
                )

    # create predictor first
    params = copy.deepcopy(params)
    params = _choose_param_value(
        main_param_name='objective',
        params=params,
        default_value=None
    )
    fobj: Optional[_LGBM_CustomObjectiveFunction] = None
    if callable(params["objective"]):
        fobj = params["objective"]
        params["objective"] = 'none'
    for alias in _ConfigAliases.get("num_iterations"):
        if alias in params:
            num_boost_round = params.pop(alias)
            _log_warning(f"Found `{alias}` in params. Will use it instead of argument")
    params["num_iterations"] = num_boost_round
    # setting early stopping via global params should be possible
    params = _choose_param_value(
        main_param_name="early_stopping_round",
        params=params,
        default_value=None
    )
    if params["early_stopping_round"] is None:
        params.pop("early_stopping_round")
    first_metric_only = params.get('first_metric_only', False)

    predictor: Optional[_InnerPredictor] = None
    if isinstance(init_model, (str, Path)):
        predictor = _InnerPredictor.from_model_file(
            model_file=init_model,
            pred_parameter=params
        )
    elif isinstance(init_model, Booster):
        predictor = _InnerPredictor.from_booster(
            booster=init_model,
            pred_parameter=dict(init_model.params, **params)
        )

    if predictor is not None:
        init_iteration = predictor.current_iteration()
    else:
        init_iteration = 0

    train_set._update_params(params) \
             ._set_predictor(predictor) \
             .set_feature_name(feature_name) \
             .set_categorical_feature(categorical_feature)

    is_valid_contain_train = False
    train_data_name = "training"
    reduced_valid_sets = []
    name_valid_sets = []
    if valid_sets is not None:
        if isinstance(valid_sets, Dataset):
            valid_sets = [valid_sets]
        if isinstance(valid_names, str):
            valid_names = [valid_names]
        for i, valid_data in enumerate(valid_sets):
            # reduce cost for prediction training data
            if valid_data is train_set:
                is_valid_contain_train = True
                if valid_names is not None:
                    train_data_name = valid_names[i]
                continue
            reduced_valid_sets.append(valid_data._update_params(params).set_reference(train_set))
            if valid_names is not None and len(valid_names) > i:
                name_valid_sets.append(valid_names[i])
            else:
                name_valid_sets.append(f'valid_{i}')
    # process callbacks
    if callbacks is None:
        callbacks_set = set()
    else:
        for i, cb in enumerate(callbacks):
            cb.__dict__.setdefault('order', i - len(callbacks))
        callbacks_set = set(callbacks)

    if "early_stopping_round" in params:
        callbacks_set.add(
            callback.early_stopping(
                stopping_rounds=params["early_stopping_round"],  # type: ignore[arg-type]
                first_metric_only=first_metric_only,
                verbose=_choose_param_value(
                    main_param_name="verbosity",
                    params=params,
                    default_value=1
                ).pop("verbosity") > 0
            )
        )

    callbacks_before_iter_set = {cb for cb in callbacks_set if getattr(cb, 'before_iteration', False)}
    callbacks_after_iter_set = callbacks_set - callbacks_before_iter_set
    callbacks_before_iter = sorted(callbacks_before_iter_set, key=attrgetter('order'))
    callbacks_after_iter = sorted(callbacks_after_iter_set, key=attrgetter('order'))

    # construct booster
    try:
        booster = Booster(params=params, train_set=train_set)
        if is_valid_contain_train:
            booster.set_train_data_name(train_data_name)
        for valid_set, name_valid_set in zip(reduced_valid_sets, name_valid_sets):
            booster.add_valid(valid_set, name_valid_set)
    finally:
        train_set._reverse_update_params()
        for valid_set in reduced_valid_sets:
            valid_set._reverse_update_params()
    booster.best_iteration = 0

    # start training
    for i in range(init_iteration, init_iteration + num_boost_round):
        for cb in callbacks_before_iter:
            cb(callback.CallbackEnv(model=booster,
                                    params=params,
                                    iteration=i,
                                    begin_iteration=init_iteration,
                                    end_iteration=init_iteration + num_boost_round,
                                    evaluation_result_list=None))

        booster.update(fobj=fobj)

        evaluation_result_list: List[_LGBM_BoosterEvalMethodResultType] = []
        # check evaluation result.
        if valid_sets is not None:
            if is_valid_contain_train:
                evaluation_result_list.extend(booster.eval_train(feval))
            evaluation_result_list.extend(booster.eval_valid(feval))
        try:
            for cb in callbacks_after_iter:
                cb(callback.CallbackEnv(model=booster,
                                        params=params,
                                        iteration=i,
                                        begin_iteration=init_iteration,
                                        end_iteration=init_iteration + num_boost_round,
                                        evaluation_result_list=evaluation_result_list))
        except callback.EarlyStopException as earlyStopException:
            booster.best_iteration = earlyStopException.best_iteration + 1
            evaluation_result_list = earlyStopException.best_score
            break
    booster.best_score = defaultdict(OrderedDict)
    for dataset_name, eval_name, score, _ in evaluation_result_list:
        booster.best_score[dataset_name][eval_name] = score
    if not keep_training_booster:
        booster.model_from_string(booster.model_to_string()).free_dataset()
    return booster


class CVBooster:
    """CVBooster in LightGBM.

    Auxiliary data structure to hold and redirect all boosters of ``cv()`` function.
    This class has the same methods as Booster class.
    All method calls, except for the following methods, are actually performed for underlying Boosters and
    then all returned results are returned in a list.

    - ``model_from_string()``
    - ``model_to_string()``
    - ``save_model()``

    Attributes
    ----------
    boosters : list of Booster
        The list of underlying fitted models.
    best_iteration : int
        The best iteration of fitted model.
    """

    def __init__(
        self,
        model_file: Optional[Union[str, Path]] = None
    ):
        """Initialize the CVBooster.

        Parameters
        ----------
        model_file : str, pathlib.Path or None, optional (default=None)
            Path to the CVBooster model file.
        """
        self.boosters: List[Booster] = []
        self.best_iteration = -1

        if model_file is not None:
            with open(model_file, "r") as file:
                self._from_dict(json.load(file))

    def _from_dict(self, models: Dict[str, Any]) -> None:
        """Load CVBooster from dict."""
        self.best_iteration = models["best_iteration"]
        self.boosters = []
        for model_str in models["boosters"]:
            self.boosters.append(Booster(model_str=model_str))

    def _to_dict(self, num_iteration: Optional[int], start_iteration: int, importance_type: str) -> Dict[str, Any]:
        """Serialize CVBooster to dict."""
        models_str = []
        for booster in self.boosters:
            models_str.append(booster.model_to_string(num_iteration=num_iteration, start_iteration=start_iteration,
                                                      importance_type=importance_type))
        return {"boosters": models_str, "best_iteration": self.best_iteration}

    def __getattr__(self, name: str) -> Callable[[Any, Any], List[Any]]:
        """Redirect methods call of CVBooster."""
        def handler_function(*args: Any, **kwargs: Any) -> List[Any]:
            """Call methods with each booster, and concatenate their results."""
            ret = []
            for booster in self.boosters:
                ret.append(getattr(booster, name)(*args, **kwargs))
            return ret
        return handler_function

    def __getstate__(self) -> Dict[str, Any]:
        return vars(self)

    def __setstate__(self, state: Dict[str, Any]) -> None:
        vars(self).update(state)

    def model_from_string(self, model_str: str) -> "CVBooster":
        """Load CVBooster from a string.

        Parameters
        ----------
        model_str : str
            Model will be loaded from this string.

        Returns
        -------
        self : CVBooster
            Loaded CVBooster object.
        """
        self._from_dict(json.loads(model_str))
        return self

    def model_to_string(
        self,
        num_iteration: Optional[int] = None,
        start_iteration: int = 0,
        importance_type: str = 'split'
    ) -> str:
        """Save CVBooster to JSON string.

        Parameters
        ----------
        num_iteration : int or None, optional (default=None)
            Index of the iteration that should be saved.
            If None, if the best iteration exists, it is saved; otherwise, all iterations are saved.
            If <= 0, all iterations are saved.
        start_iteration : int, optional (default=0)
            Start index of the iteration that should be saved.
        importance_type : str, optional (default="split")
            What type of feature importance should be saved.
            If "split", result contains numbers of times the feature is used in a model.
            If "gain", result contains total gains of splits which use the feature.

        Returns
        -------
        str_repr : str
            JSON string representation of CVBooster.
        """
        return json.dumps(self._to_dict(num_iteration, start_iteration, importance_type))

    def save_model(
        self,
        filename: Union[str, Path],
        num_iteration: Optional[int] = None,
        start_iteration: int = 0,
        importance_type: str = 'split'
    ) -> "CVBooster":
        """Save CVBooster to a file as JSON text.

        Parameters
        ----------
        filename : str or pathlib.Path
            Filename to save CVBooster.
        num_iteration : int or None, optional (default=None)
            Index of the iteration that should be saved.
            If None, if the best iteration exists, it is saved; otherwise, all iterations are saved.
            If <= 0, all iterations are saved.
        start_iteration : int, optional (default=0)
            Start index of the iteration that should be saved.
        importance_type : str, optional (default="split")
            What type of feature importance should be saved.
            If "split", result contains numbers of times the feature is used in a model.
            If "gain", result contains total gains of splits which use the feature.

        Returns
        -------
        self : CVBooster
            Returns self.
        """
        with open(filename, "w") as file:
            json.dump(self._to_dict(num_iteration, start_iteration, importance_type), file)

        return self


def _make_n_folds(
    full_data: Dataset,
    folds: Optional[Union[Iterable[Tuple[np.ndarray, np.ndarray]], _LGBMBaseCrossValidator]],
    nfold: int,
    params: Dict[str, Any],
    seed: int,
    fpreproc: Optional[_LGBM_PreprocFunction],
    stratified: bool,
    shuffle: bool,
    eval_train_metric: bool
) -> CVBooster:
    """Make a n-fold list of Booster from random indices."""
    full_data = full_data.construct()
    num_data = full_data.num_data()
    if folds is not None:
        if not hasattr(folds, '__iter__') and not hasattr(folds, 'split'):
            raise AttributeError("folds should be a generator or iterator of (train_idx, test_idx) tuples "
                                 "or scikit-learn splitter object with split method")
        if hasattr(folds, 'split'):
            group_info = full_data.get_group()
            if group_info is not None:
                group_info = np.array(group_info, dtype=np.int32, copy=False)
                flatted_group = np.repeat(range(len(group_info)), repeats=group_info)
            else:
                flatted_group = np.zeros(num_data, dtype=np.int32)
            folds = folds.split(X=np.empty(num_data), y=full_data.get_label(), groups=flatted_group)
    else:
        if any(params.get(obj_alias, "") in {"lambdarank", "rank_xendcg", "xendcg",
                                             "xe_ndcg", "xe_ndcg_mart", "xendcg_mart"}
               for obj_alias in _ConfigAliases.get("objective")):
            if not SKLEARN_INSTALLED:
                raise LightGBMError('scikit-learn is required for ranking cv')
            # ranking task, split according to groups
            group_info = np.array(full_data.get_group(), dtype=np.int32, copy=False)
            flatted_group = np.repeat(range(len(group_info)), repeats=group_info)
            group_kfold = _LGBMGroupKFold(n_splits=nfold)
            folds = group_kfold.split(X=np.empty(num_data), groups=flatted_group)
        elif stratified:
            if not SKLEARN_INSTALLED:
                raise LightGBMError('scikit-learn is required for stratified cv')
            skf = _LGBMStratifiedKFold(n_splits=nfold, shuffle=shuffle, random_state=seed)
            folds = skf.split(X=np.empty(num_data), y=full_data.get_label())
        else:
            if shuffle:
                randidx = np.random.RandomState(seed).permutation(num_data)
            else:
                randidx = np.arange(num_data)
            kstep = int(num_data / nfold)
            test_id = [randidx[i: i + kstep] for i in range(0, num_data, kstep)]
            train_id = [np.concatenate([test_id[i] for i in range(nfold) if k != i]) for k in range(nfold)]
            folds = zip(train_id, test_id)

    ret = CVBooster()
    for train_idx, test_idx in folds:
        train_set = full_data.subset(sorted(train_idx))
        valid_set = full_data.subset(sorted(test_idx))
        # run preprocessing on the data set if needed
        if fpreproc is not None:
            train_set, valid_set, tparam = fpreproc(train_set, valid_set, params.copy())
        else:
            tparam = params
        booster_for_fold = Booster(tparam, train_set)
        if eval_train_metric:
            booster_for_fold.add_valid(train_set, 'train')
        booster_for_fold.add_valid(valid_set, 'valid')
        ret.boosters.append(booster_for_fold)
    return ret


def _agg_cv_result(
    raw_results: List[List[Tuple[str, str, float, bool]]]
) -> List[Tuple[str, str, float, bool, float]]:
    """Aggregate cross-validation results."""
    cvmap: Dict[str, List[float]] = OrderedDict()
    metric_type: Dict[str, bool] = {}
    for one_result in raw_results:
        for one_line in one_result:
            key = f"{one_line[0]} {one_line[1]}"
            metric_type[key] = one_line[3]
            cvmap.setdefault(key, [])
            cvmap[key].append(one_line[2])
    return [('cv_agg', k, np.mean(v), metric_type[k], np.std(v)) for k, v in cvmap.items()]


def cv(
    params: Dict[str, Any],
    train_set: Dataset,
    num_boost_round: int = 100,
    folds: Optional[Union[Iterable[Tuple[np.ndarray, np.ndarray]], _LGBMBaseCrossValidator]] = None,
    nfold: int = 5,
    stratified: bool = True,
    shuffle: bool = True,
    metrics: Optional[Union[str, List[str]]] = None,
    feval: Optional[Union[_LGBM_CustomMetricFunction, List[_LGBM_CustomMetricFunction]]] = None,
    init_model: Optional[Union[str, Path, Booster]] = None,
    feature_name: _LGBM_FeatureNameConfiguration = 'auto',
    categorical_feature: _LGBM_CategoricalFeatureConfiguration = 'auto',
    fpreproc: Optional[_LGBM_PreprocFunction] = None,
    seed: int = 0,
    callbacks: Optional[List[Callable]] = None,
    eval_train_metric: bool = False,
    return_cvbooster: bool = False
) -> Dict[str, Union[List[float], CVBooster]]:
    """Perform the cross-validation with given parameters.

    Parameters
    ----------
    params : dict
        Parameters for training. Values passed through ``params`` take precedence over those
        supplied via arguments.
    train_set : Dataset
        Data to be trained on.
    num_boost_round : int, optional (default=100)
        Number of boosting iterations.
    folds : generator or iterator of (train_idx, test_idx) tuples, scikit-learn splitter object or None, optional (default=None)
        If generator or iterator, it should yield the train and test indices for each fold.
        If object, it should be one of the scikit-learn splitter classes
        (https://scikit-learn.org/stable/modules/classes.html#splitter-classes)
        and have ``split`` method.
        This argument has highest priority over other data split arguments.
    nfold : int, optional (default=5)
        Number of folds in CV.
    stratified : bool, optional (default=True)
        Whether to perform stratified sampling.
    shuffle : bool, optional (default=True)
        Whether to shuffle before splitting data.
    metrics : str, list of str, or None, optional (default=None)
        Evaluation metrics to be monitored while CV.
        If not None, the metric in ``params`` will be overridden.
    feval : callable, list of callable, or None, optional (default=None)
        Customized evaluation function.
        Each evaluation function should accept two parameters: preds, eval_data,
        and return (eval_name, eval_result, is_higher_better) or list of such tuples.

            preds : numpy 1-D array or numpy 2-D array (for multi-class task)
                The predicted values.
                For multi-class task, preds are numpy 2-D array of shape = [n_samples, n_classes].
                If custom objective function is used, predicted values are returned before any transformation,
                e.g. they are raw margin instead of probability of positive class for binary task in this case.
            eval_data : Dataset
                A ``Dataset`` to evaluate.
            eval_name : str
                The name of evaluation function (without whitespace).
            eval_result : float
                The eval result.
            is_higher_better : bool
                Is eval result higher better, e.g. AUC is ``is_higher_better``.

        To ignore the default metric corresponding to the used objective,
        set ``metrics`` to the string ``"None"``.
    init_model : str, pathlib.Path, Booster or None, optional (default=None)
        Filename of LightGBM model or Booster instance used for continue training.
    feature_name : list of str, or 'auto', optional (default="auto")
        Feature names.
        If 'auto' and data is pandas DataFrame, data columns names are used.
    categorical_feature : list of str or int, or 'auto', optional (default="auto")
        Categorical features.
        If list of int, interpreted as indices.
        If list of str, interpreted as feature names (need to specify ``feature_name`` as well).
        If 'auto' and data is pandas DataFrame, pandas unordered categorical columns are used.
        All values in categorical features will be cast to int32 and thus should be less than int32 max value (2147483647).
        Large values could be memory consuming. Consider using consecutive integers starting from zero.
        All negative values in categorical features will be treated as missing values.
        The output cannot be monotonically constrained with respect to a categorical feature.
        Floating point numbers in categorical features will be rounded towards 0.
    fpreproc : callable or None, optional (default=None)
        Preprocessing function that takes (dtrain, dtest, params)
        and returns transformed versions of those.
    seed : int, optional (default=0)
        Seed used to generate the folds (passed to numpy.random.seed).
    callbacks : list of callable, or None, optional (default=None)
        List of callback functions that are applied at each iteration.
        See Callbacks in Python API for more information.
    eval_train_metric : bool, optional (default=False)
        Whether to display the train metric in progress.
        The score of the metric is calculated again after each training step, so there is some impact on performance.
    return_cvbooster : bool, optional (default=False)
        Whether to return Booster models trained on each fold through ``CVBooster``.

    Note
    ----
    A custom objective function can be provided for the ``objective`` parameter.
    It should accept two parameters: preds, train_data and return (grad, hess).

        preds : numpy 1-D array or numpy 2-D array (for multi-class task)
            The predicted values.
            Predicted values are returned before any transformation,
            e.g. they are raw margin instead of probability of positive class for binary task.
        train_data : Dataset
            The training dataset.
        grad : numpy 1-D array or numpy 2-D array (for multi-class task)
            The value of the first order derivative (gradient) of the loss
            with respect to the elements of preds for each sample point.
        hess : numpy 1-D array or numpy 2-D array (for multi-class task)
            The value of the second order derivative (Hessian) of the loss
            with respect to the elements of preds for each sample point.

    For multi-class task, preds are numpy 2-D array of shape = [n_samples, n_classes],
    and grad and hess should be returned in the same format.

    Returns
    -------
    eval_results : dict
        History of evaluation results of each metric.
        The dictionary has the following format:
        {'valid metric1-mean': [values], 'valid metric1-stdv': [values],
        'valid metric2-mean': [values], 'valid metric2-stdv': [values],
        ...}.
        If ``return_cvbooster=True``, also returns trained boosters wrapped in a ``CVBooster`` object via ``cvbooster`` key.
        If ``eval_train_metric=True``, also returns the train metric history.
        In this case, the dictionary has the following format:
        {'train metric1-mean': [values], 'valid metric1-mean': [values],
        'train metric2-mean': [values], 'valid metric2-mean': [values],
        ...}.
    """
    if not isinstance(train_set, Dataset):
        raise TypeError(f"cv() only accepts Dataset object, train_set has type '{type(train_set).__name__}'.")

    if num_boost_round <= 0:
        raise ValueError(f"num_boost_round must be greater than 0. Got {num_boost_round}.")

    params = copy.deepcopy(params)
    params = _choose_param_value(
        main_param_name='objective',
        params=params,
        default_value=None
    )
    fobj: Optional[_LGBM_CustomObjectiveFunction] = None
    if callable(params["objective"]):
        fobj = params["objective"]
        params["objective"] = 'none'
    for alias in _ConfigAliases.get("num_iterations"):
        if alias in params:
            _log_warning(f"Found '{alias}' in params. Will use it instead of 'num_boost_round' argument")
            num_boost_round = params.pop(alias)
    params["num_iterations"] = num_boost_round
    # setting early stopping via global params should be possible
    params = _choose_param_value(
        main_param_name="early_stopping_round",
        params=params,
        default_value=None
    )
    if params["early_stopping_round"] is None:
        params.pop("early_stopping_round")
    first_metric_only = params.get('first_metric_only', False)

    if isinstance(init_model, (str, Path)):
        predictor = _InnerPredictor.from_model_file(
            model_file=init_model,
            pred_parameter=params
        )
    elif isinstance(init_model, Booster):
        predictor = _InnerPredictor.from_booster(
            booster=init_model,
            pred_parameter=dict(init_model.params, **params)
        )
    else:
        predictor = None

    if metrics is not None:
        for metric_alias in _ConfigAliases.get("metric"):
            params.pop(metric_alias, None)
        params['metric'] = metrics

    train_set._update_params(params) \
             ._set_predictor(predictor) \
             .set_feature_name(feature_name) \
             .set_categorical_feature(categorical_feature)

    results = defaultdict(list)
    cvfolds = _make_n_folds(full_data=train_set, folds=folds, nfold=nfold,
                            params=params, seed=seed, fpreproc=fpreproc,
                            stratified=stratified, shuffle=shuffle,
                            eval_train_metric=eval_train_metric)

    # setup callbacks
    if callbacks is None:
        callbacks_set = set()
    else:
        for i, cb in enumerate(callbacks):
            cb.__dict__.setdefault('order', i - len(callbacks))
        callbacks_set = set(callbacks)

    if "early_stopping_round" in params:
        callbacks_set.add(
            callback.early_stopping(
                stopping_rounds=params["early_stopping_round"],  # type: ignore[arg-type]
                first_metric_only=first_metric_only,
                verbose=_choose_param_value(
                    main_param_name="verbosity",
                    params=params,
                    default_value=1
                ).pop("verbosity") > 0
            )
        )

    callbacks_before_iter_set = {cb for cb in callbacks_set if getattr(cb, 'before_iteration', False)}
    callbacks_after_iter_set = callbacks_set - callbacks_before_iter_set
    callbacks_before_iter = sorted(callbacks_before_iter_set, key=attrgetter('order'))
    callbacks_after_iter = sorted(callbacks_after_iter_set, key=attrgetter('order'))

    for i in range(num_boost_round):
        for cb in callbacks_before_iter:
            cb(callback.CallbackEnv(model=cvfolds,
                                    params=params,
                                    iteration=i,
                                    begin_iteration=0,
                                    end_iteration=num_boost_round,
                                    evaluation_result_list=None))
        cvfolds.update(fobj=fobj)  # type: ignore[call-arg]
        res = _agg_cv_result(cvfolds.eval_valid(feval))  # type: ignore[call-arg]
        for _, key, mean, _, std in res:
            results[f'{key}-mean'].append(mean)
            results[f'{key}-stdv'].append(std)
        try:
            for cb in callbacks_after_iter:
                cb(callback.CallbackEnv(model=cvfolds,
                                        params=params,
                                        iteration=i,
                                        begin_iteration=0,
                                        end_iteration=num_boost_round,
                                        evaluation_result_list=res))
        except callback.EarlyStopException as earlyStopException:
            cvfolds.best_iteration = earlyStopException.best_iteration + 1
            for bst in cvfolds.boosters:
                bst.best_iteration = cvfolds.best_iteration
            for k in results:
                results[k] = results[k][:cvfolds.best_iteration]
            break

    if return_cvbooster:
        results['cvbooster'] = cvfolds  # type: ignore[assignment]

    return dict(results)
