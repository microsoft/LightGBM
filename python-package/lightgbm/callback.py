# coding: utf-8
# pylint: disable = invalid-name, W0105, C0301
from __future__ import absolute_import

import collections


class EarlyStopException(Exception):
    """Exception of early stopping.
    Parameters
    ----------
    best_iteration : int
        The best iteration stopped.
    """
    def __init__(self, best_iteration):
        super(EarlyStopException, self).__init__()
        self.best_iteration = best_iteration


# Callback environment used by callbacks
CallbackEnv = collections.namedtuple(
    "LightGBMCallbackEnv",
    ["model",
     "params",
     "iteration",
     "begin_iteration",
     "end_iteration",
     "evaluation_result_list"])


def _format_eval_result(value, show_stdv=True):
    """format metric string"""
    if len(value) == 4:
        return '%s\'s %s: %g' % (value[0], value[1], value[2])
    elif len(value) == 5:
        if show_stdv:
            return '%s\'s %s: %g + %g' % (value[0], value[1], value[2], value[4])
        else:
            return '%s\'s %s: %g' % (value[0], value[1], value[2])
    else:
        raise ValueError("Wrong metric value")


def print_evaluation(period=1, show_stdv=True):
    """Create a callback that print evaluation result.

    Parameters
    ----------
    period : int
        The period to log the evaluation results

    show_stdv : bool, optional
        Whether show stdv if provided

    Returns
    -------
    callback : function
        A callback that print evaluation every period iterations.
    """
    def callback(env):
        """internal function"""
        if period > 0 and env.evaluation_result_list and (env.iteration + 1) % period == 0:
            result = '\t'.join([_format_eval_result(x, show_stdv) for x in env.evaluation_result_list])
            print('[%d]\t%s' % (env.iteration + 1, result))
    callback.order = 10
    return callback


def record_evaluation(eval_result):
    """Create a call back that records the evaluation history into eval_result.

    Parameters
    ----------
    eval_result : dict
       A dictionary to store the evaluation results.

    Returns
    -------
    callback : function
        The requested callback function.
    """
    if not isinstance(eval_result, dict):
        raise TypeError('Eval_result should be a dictionary')
    eval_result.clear()

    def init(env):
        """internal function"""
        for data_name, _, _, _ in env.evaluation_result_list:
            eval_result.setdefault(data_name, collections.defaultdict(list))

    def callback(env):
        """internal function"""
        if not eval_result:
            init(env)
        for data_name, eval_name, result, _ in env.evaluation_result_list:
            eval_result[data_name][eval_name].append(result)
    callback.order = 20
    return callback


def reset_parameter(**kwargs):
    """Reset parameter after first iteration

    NOTE: the initial parameter will still take in-effect on first iteration.

    Parameters
    ----------
    **kwargs: value should be list or function
        List of parameters for each boosting round
        or a customized function that calculates learning_rate in terms of
        current number of round (e.g. yields learning rate decay)
        - list l: parameter = l[current_round]
        - function f: parameter = f(current_round)
    Returns
    -------
    callback : function
        The requested callback function.
    """
    def callback(env):
        """internal function"""
        new_parameters = {}
        for key, value in kwargs.items():
            if key in ['num_class', 'boosting_type', 'metric']:
                raise RuntimeError("cannot reset {} during training".format(repr(key)))
            if isinstance(value, list):
                if len(value) != env.end_iteration - env.begin_iteration:
                    raise ValueError("Length of list {} has to equal to 'num_boost_round'.".format(repr(key)))
                new_param = value[env.iteration - env.begin_iteration]
            else:
                new_param = value(env.iteration - env.begin_iteration)
            if new_param != env.params.get(key, None):
                new_parameters[key] = new_param
        if new_parameters:
            env.model.reset_parameter(new_parameters)
            env.params.update(new_parameters)
    callback.before_iteration = True
    callback.order = 10
    return callback


def early_stopping(stopping_rounds, verbose=True):
    """Create a callback that activates early stopping.
    Activates early stopping.
    Requires at least one validation data and one metric
    If there's more than one, will check all of them

    Parameters
    ----------
    stopping_rounds : int
       The stopping rounds before the trend occur.

    verbose : optional, bool
        Whether to print message about early stopping information.

    Returns
    -------
    callback : function
        The requested callback function.
    """
    factor_to_bigger_better = {}
    best_score = {}
    best_iter = {}
    best_msg = {}

    def init(env):
        """internal function"""
        if not env.evaluation_result_list:
            raise ValueError('For early stopping, at least one dataset or eval metric is required for evaluation')

        if verbose:
            msg = "Train until valid scores didn't improve in {} rounds."
            print(msg.format(stopping_rounds))

        for i in range(len(env.evaluation_result_list)):
            best_score[i] = float('-inf')
            best_iter[i] = 0
            if verbose:
                best_msg[i] = ""
            factor_to_bigger_better[i] = 1.0 if env.evaluation_result_list[i][3] else -1.0

    def callback(env):
        """internal function"""
        if not best_score:
            init(env)
        for i in range(len(env.evaluation_result_list)):
            score = env.evaluation_result_list[i][2] * factor_to_bigger_better[i]
            if score > best_score[i]:
                best_score[i] = score
                best_iter[i] = env.iteration
                if verbose:
                    best_msg[i] = '[%d]\t%s' % (
                        env.iteration + 1, '\t'.join(
                            [_format_eval_result(x) for x in env.evaluation_result_list]
                        )
                    )
            else:
                if env.iteration - best_iter[i] >= stopping_rounds:
                    env.model.set_attr(best_iteration=str(best_iter[i]))
                    if verbose:
                        print('Early stopping, best iteration is:')
                        print(best_msg[i])
                    raise EarlyStopException(best_iter[i])
    callback.order = 30
    return callback
