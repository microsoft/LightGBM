# coding: utf-8
"""Callbacks library."""
import collections
from functools import partial
from typing import Any, Callable, Dict, List, Tuple, Union

from .basic import _ConfigAliases, _LGBM_BoosterEvalMethodResultType, _log_info, _log_warning

_EvalResultTuple = Union[
    List[_LGBM_BoosterEvalMethodResultType],
    List[Tuple[str, str, float, bool, float]]
]


class EarlyStopException(Exception):
    """Exception of early stopping."""

    def __init__(self, best_iteration: int, best_score: _EvalResultTuple) -> None:
        """Create early stopping exception.

        Parameters
        ----------
        best_iteration : int
            The best iteration stopped.
        best_score : list of (eval_name, metric_name, eval_result, is_higher_better) tuple or (eval_name, metric_name, eval_result, is_higher_better, stdv) tuple
            Scores for each metric, on each validation set, as of the best iteration.
        """
        super().__init__()
        self.best_iteration = best_iteration
        self.best_score = best_score


# Callback environment used by callbacks
CallbackEnv = collections.namedtuple(
    "CallbackEnv",
    ["model",
     "params",
     "iteration",
     "begin_iteration",
     "end_iteration",
     "evaluation_result_list"])


def _format_eval_result(value: _EvalResultTuple, show_stdv: bool = True) -> str:
    """Format metric string."""
    if len(value) == 4:
        return f"{value[0]}'s {value[1]}: {value[2]:g}"
    elif len(value) == 5:
        if show_stdv:
            return f"{value[0]}'s {value[1]}: {value[2]:g} + {value[4]:g}"
        else:
            return f"{value[0]}'s {value[1]}: {value[2]:g}"
    else:
        raise ValueError("Wrong metric value")


class _LogEvaluationCallback:
    """Internal log evaluation callable class."""

    def __init__(self, period: int = 1, show_stdv: bool = True) -> None:
        self.order = 10
        self.before_iteration = False

        self.period = period
        self.show_stdv = show_stdv

    def __call__(self, env: CallbackEnv) -> None:
        if self.period > 0 and env.evaluation_result_list and (env.iteration + 1) % self.period == 0:
            result = '\t'.join([_format_eval_result(x, self.show_stdv) for x in env.evaluation_result_list])
            _log_info(f'[{env.iteration + 1}]\t{result}')


def log_evaluation(period: int = 1, show_stdv: bool = True) -> _LogEvaluationCallback:
    """Create a callback that logs the evaluation results.

    By default, standard output resource is used.
    Use ``register_logger()`` function to register a custom logger.

    Note
    ----
    Requires at least one validation data.

    Parameters
    ----------
    period : int, optional (default=1)
        The period to log the evaluation results.
        The last boosting stage or the boosting stage found by using ``early_stopping`` callback is also logged.
    show_stdv : bool, optional (default=True)
        Whether to log stdv (if provided).

    Returns
    -------
    callback : _LogEvaluationCallback
        The callback that logs the evaluation results every ``period`` boosting iteration(s).
    """
    return _LogEvaluationCallback(period=period, show_stdv=show_stdv)


class _RecordEvaluationCallback:
    """Internal record evaluation callable class."""

    def __init__(self, eval_result: Dict[str, Dict[str, List[Any]]]) -> None:
        self.order = 20
        self.before_iteration = False

        if not isinstance(eval_result, dict):
            raise TypeError('eval_result should be a dictionary')
        self.eval_result = eval_result

    def _init(self, env: CallbackEnv) -> None:
        self.eval_result.clear()
        for item in env.evaluation_result_list:
            if len(item) == 4:  # regular train
                data_name, eval_name = item[:2]
            else:  # cv
                data_name, eval_name = item[1].split()
            self.eval_result.setdefault(data_name, collections.OrderedDict())
            if len(item) == 4:
                self.eval_result[data_name].setdefault(eval_name, [])
            else:
                self.eval_result[data_name].setdefault(f'{eval_name}-mean', [])
                self.eval_result[data_name].setdefault(f'{eval_name}-stdv', [])

    def __call__(self, env: CallbackEnv) -> None:
        if env.iteration == env.begin_iteration:
            self._init(env)
        for item in env.evaluation_result_list:
            if len(item) == 4:
                data_name, eval_name, result = item[:3]
                self.eval_result[data_name][eval_name].append(result)
            else:
                data_name, eval_name = item[1].split()
                res_mean = item[2]
                res_stdv = item[4]
                self.eval_result[data_name][f'{eval_name}-mean'].append(res_mean)
                self.eval_result[data_name][f'{eval_name}-stdv'].append(res_stdv)


def record_evaluation(eval_result: Dict[str, Dict[str, List[Any]]]) -> Callable:
    """Create a callback that records the evaluation history into ``eval_result``.

    Parameters
    ----------
    eval_result : dict
        Dictionary used to store all evaluation results of all validation sets.
        This should be initialized outside of your call to ``record_evaluation()`` and should be empty.
        Any initial contents of the dictionary will be deleted.

        .. rubric:: Example

        With two validation sets named 'eval' and 'train', and one evaluation metric named 'logloss'
        this dictionary after finishing a model training process will have the following structure:

        .. code-block::

            {
             'train':
                 {
                  'logloss': [0.48253, 0.35953, ...]
                 },
             'eval':
                 {
                  'logloss': [0.480385, 0.357756, ...]
                 }
            }

    Returns
    -------
    callback : _RecordEvaluationCallback
        The callback that records the evaluation history into the passed dictionary.
    """
    return _RecordEvaluationCallback(eval_result=eval_result)


class _ResetParameterCallback:
    """Internal reset parameter callable class."""

    def __init__(self, **kwargs: Union[list, Callable]) -> None:
        self.order = 10
        self.before_iteration = True

        self.kwargs = kwargs

    def __call__(self, env: CallbackEnv) -> None:
        new_parameters = {}
        for key, value in self.kwargs.items():
            if isinstance(value, list):
                if len(value) != env.end_iteration - env.begin_iteration:
                    raise ValueError(f"Length of list {key!r} has to be equal to 'num_boost_round'.")
                new_param = value[env.iteration - env.begin_iteration]
            elif callable(value):
                new_param = value(env.iteration - env.begin_iteration)
            else:
                raise ValueError("Only list and callable values are supported "
                                 "as a mapping from boosting round index to new parameter value.")
            if new_param != env.params.get(key, None):
                new_parameters[key] = new_param
        if new_parameters:
            env.model.reset_parameter(new_parameters)
            env.params.update(new_parameters)


def reset_parameter(**kwargs: Union[list, Callable]) -> Callable:
    """Create a callback that resets the parameter after the first iteration.

    .. note::

        The initial parameter will still take in-effect on first iteration.

    Parameters
    ----------
    **kwargs : value should be list or callable
        List of parameters for each boosting round
        or a callable that calculates the parameter in terms of
        current number of round (e.g. yields learning rate decay).
        If list lst, parameter = lst[current_round].
        If callable func, parameter = func(current_round).

    Returns
    -------
    callback : _ResetParameterCallback
        The callback that resets the parameter after the first iteration.
    """
    return _ResetParameterCallback(**kwargs)


class _EarlyStoppingCallback:
    """Internal early stopping callable class."""

    def __init__(
        self,
        stopping_rounds: int,
        first_metric_only: bool = False,
        verbose: bool = True,
        min_delta: Union[float, List[float]] = 0.0
    ) -> None:
        self.order = 30
        self.before_iteration = False

        self.stopping_rounds = stopping_rounds
        self.first_metric_only = first_metric_only
        self.verbose = verbose
        self.min_delta = min_delta

        self.enabled = True
        self._reset_storages()

    def _reset_storages(self) -> None:
        self.best_score = []
        self.best_iter = []
        self.best_score_list = []
        self.cmp_op = []
        self.first_metric = ''

    def _gt_delta(self, curr_score: float, best_score: float, delta: float) -> bool:
        return curr_score > best_score + delta

    def _lt_delta(self, curr_score: float, best_score: float, delta: float) -> bool:
        return curr_score < best_score - delta

    def _is_train_set(self, ds_name: str, eval_name: str, train_name: str) -> bool:
        return (ds_name == "cv_agg" and eval_name == "train") or ds_name == train_name

    def _init(self, env: CallbackEnv) -> None:
        is_dart = any(env.params.get(alias, "") == 'dart' for alias in _ConfigAliases.get("boosting"))
        only_train_set = (
            len(env.evaluation_result_list) == 1
            and self._is_train_set(
                ds_name=env.evaluation_result_list[0][0],
                eval_name=env.evaluation_result_list[0][1].split(" ")[0],
                train_name=env.model._train_data_name)
        )
        self.enabled = not is_dart and not only_train_set
        if not self.enabled:
            if is_dart:
                _log_warning('Early stopping is not available in dart mode')
            elif only_train_set:
                _log_warning('Only training set found, disabling early stopping.')
            return
        if not env.evaluation_result_list:
            raise ValueError('For early stopping, '
                             'at least one dataset and eval metric is required for evaluation')

        if self.stopping_rounds <= 0:
            raise ValueError("stopping_rounds should be greater than zero.")

        if self.verbose:
            _log_info(f"Training until validation scores don't improve for {self.stopping_rounds} rounds")

        self._reset_storages()

        n_metrics = len(set(m[1] for m in env.evaluation_result_list))
        n_datasets = len(env.evaluation_result_list) // n_metrics
        if isinstance(self.min_delta, list):
            if not all(t >= 0 for t in self.min_delta):
                raise ValueError('Values for early stopping min_delta must be non-negative.')
            if len(self.min_delta) == 0:
                if self.verbose:
                    _log_info('Disabling min_delta for early stopping.')
                deltas = [0.0] * n_datasets * n_metrics
            elif len(self.min_delta) == 1:
                if self.verbose:
                    _log_info(f'Using {self.min_delta[0]} as min_delta for all metrics.')
                deltas = self.min_delta * n_datasets * n_metrics
            else:
                if len(self.min_delta) != n_metrics:
                    raise ValueError('Must provide a single value for min_delta or as many as metrics.')
                if self.first_metric_only and self.verbose:
                    _log_info(f'Using only {self.min_delta[0]} as early stopping min_delta.')
                deltas = self.min_delta * n_datasets
        else:
            if self.min_delta < 0:
                raise ValueError('Early stopping min_delta must be non-negative.')
            if self.min_delta > 0 and n_metrics > 1 and not self.first_metric_only and self.verbose:
                _log_info(f'Using {self.min_delta} as min_delta for all metrics.')
            deltas = [self.min_delta] * n_datasets * n_metrics

        # split is needed for "<dataset type> <metric>" case (e.g. "train l1")
        self.first_metric = env.evaluation_result_list[0][1].split(" ")[-1]
        for eval_ret, delta in zip(env.evaluation_result_list, deltas):
            self.best_iter.append(0)
            self.best_score_list.append(None)
            if eval_ret[3]:  # greater is better
                self.best_score.append(float('-inf'))
                self.cmp_op.append(partial(self._gt_delta, delta=delta))
            else:
                self.best_score.append(float('inf'))
                self.cmp_op.append(partial(self._lt_delta, delta=delta))

    def _final_iteration_check(self, env: CallbackEnv, eval_name_splitted: List[str], i: int) -> None:
        if env.iteration == env.end_iteration - 1:
            if self.verbose:
                best_score_str = '\t'.join([_format_eval_result(x) for x in self.best_score_list[i]])
                _log_info('Did not meet early stopping. '
                          f'Best iteration is:\n[{self.best_iter[i] + 1}]\t{best_score_str}')
                if self.first_metric_only:
                    _log_info(f"Evaluated only: {eval_name_splitted[-1]}")
            raise EarlyStopException(self.best_iter[i], self.best_score_list[i])

    def __call__(self, env: CallbackEnv) -> None:
        if env.iteration == env.begin_iteration:
            self._init(env)
        if not self.enabled:
            return
        for i in range(len(env.evaluation_result_list)):
            score = env.evaluation_result_list[i][2]
            if self.best_score_list[i] is None or self.cmp_op[i](score, self.best_score[i]):
                self.best_score[i] = score
                self.best_iter[i] = env.iteration
                self.best_score_list[i] = env.evaluation_result_list
            # split is needed for "<dataset type> <metric>" case (e.g. "train l1")
            eval_name_splitted = env.evaluation_result_list[i][1].split(" ")
            if self.first_metric_only and self.first_metric != eval_name_splitted[-1]:
                continue  # use only the first metric for early stopping
            if self._is_train_set(env.evaluation_result_list[i][0], eval_name_splitted[0], env.model._train_data_name):
                continue  # train data for lgb.cv or sklearn wrapper (underlying lgb.train)
            elif env.iteration - self.best_iter[i] >= self.stopping_rounds:
                if self.verbose:
                    eval_result_str = '\t'.join([_format_eval_result(x) for x in self.best_score_list[i]])
                    _log_info(f"Early stopping, best iteration is:\n[{self.best_iter[i] + 1}]\t{eval_result_str}")
                    if self.first_metric_only:
                        _log_info(f"Evaluated only: {eval_name_splitted[-1]}")
                raise EarlyStopException(self.best_iter[i], self.best_score_list[i])
            self._final_iteration_check(env, eval_name_splitted, i)


def early_stopping(stopping_rounds: int, first_metric_only: bool = False, verbose: bool = True, min_delta: Union[float, List[float]] = 0.0) -> _EarlyStoppingCallback:
    """Create a callback that activates early stopping.

    Activates early stopping.
    The model will train until the validation score doesn't improve by at least ``min_delta``.
    Validation score needs to improve at least every ``stopping_rounds`` round(s)
    to continue training.
    Requires at least one validation data and one metric.
    If there's more than one, will check all of them. But the training data is ignored anyway.
    To check only the first metric set ``first_metric_only`` to True.
    The index of iteration that has the best performance will be saved in the ``best_iteration`` attribute of a model.

    Parameters
    ----------
    stopping_rounds : int
        The possible number of rounds without the trend occurrence.
    first_metric_only : bool, optional (default=False)
        Whether to use only the first metric for early stopping.
    verbose : bool, optional (default=True)
        Whether to log message with early stopping information.
        By default, standard output resource is used.
        Use ``register_logger()`` function to register a custom logger.
    min_delta : float or list of float, optional (default=0.0)
        Minimum improvement in score to keep training.
        If float, this single value is used for all metrics.
        If list, its length should match the total number of metrics.

    Returns
    -------
    callback : _EarlyStoppingCallback
        The callback that activates early stopping.
    """
    return _EarlyStoppingCallback(stopping_rounds=stopping_rounds, first_metric_only=first_metric_only, verbose=verbose, min_delta=min_delta)
