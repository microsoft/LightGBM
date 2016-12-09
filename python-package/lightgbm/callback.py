# coding: utf-8
# pylint: disable = invalid-name, W0105, C0301
from __future__ import absolute_import
import collections
import inspect

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
     "cvfolds",
     "iteration",
     "begin_iteration",
     "end_iteration",
     "evaluation_result_list"])

def _format_eval_result(value, show_stdv=True):
    """format metric string"""
    if len(value) == 4:
        return '%s\'s %s:%g' % (value[0], value[1], value[2])
    elif len(value) == 5:
        if show_stdv:
            return '%s\'s %s:%g+%g' % (value[0], value[1], value[2], value[4])
        else:
            return '%s\'s %s:%g' % (value[0], value[1], value[2])
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
        if not env.evaluation_result_list or period <= 0:
            return
        if (env.iteration + 1) % period == 0:
            result = '\t'.join([_format_eval_result(x, show_stdv) \
                for x in env.evaluation_result_list])
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


def reset_learning_rate(learning_rates):
    """Reset learning rate after first iteration

    NOTE: the initial learning rate will still take in-effect on first iteration.

    Parameters
    ----------
    learning_rates: list or function
        List of learning rate for each boosting round \
        or a customized function that calculates learning_rate in terms of \
        current number of round and the total number of boosting round \
        (e.g. yields learning rate decay)
        - list l: learning_rate = l[current_round]
        - function f: learning_rate = f(current_round, total_boost_round) \
                   or learning_rate = f(current_round)
    Returns
    -------
    callback : function
        The requested callback function.
    """
    def callback(env):
        """internal function"""
        if isinstance(learning_rates, list):
            if len(learning_rates) != env.end_iteration - env.begin_iteration:
                raise ValueError("Length of list 'learning_rates' has to equal to 'num_boost_round'.")
            env.model.reset_parameter({'learning_rate':learning_rates[env.iteration]})
        else:
            argc = len(inspect.getargspec(learning_rates).args)
            if argc is 1:
                env.model.reset_parameter({"learning_rate": learning_rates(env.iteration - env.begin_iteration)})
            elif argc is 2:
                env.model.reset_parameter({"learning_rate": \
                    learning_rates(env.iteration - env.begin_iteration, env.end_iteration - env.begin_iteration)})
            else:
                raise ValueError("Self-defined function 'learning_rates' should have 1 or 2 arguments")
    callback.before_iteration = True
    callback.order = 10
    return callback


def early_stop(stopping_rounds, verbose=True):
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
            raise ValueError('For early stopping, at least one dataset is required for evaluation')

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
                    best_msg[i] = '[%d]\t%s' % (env.iteration + 1, \
                        '\t'.join([_format_eval_result(x) for x in env.evaluation_result_list]))
            else:
                if env.iteration - best_iter[i] >= stopping_rounds:
                    if env.model is not None:
                        env.model.set_attr(best_iteration=str(best_iter[i]))
                    if verbose:
                        print('Early stopping, best iteration is:')
                        print(best_msg[i])
                    raise EarlyStopException(best_iter[i])
    callback.order = 30
    return callback
