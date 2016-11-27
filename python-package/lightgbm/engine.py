"""Training Library containing training routines of LightGBM."""
from __future__ import absolute_import

import collections
import numpy as np
from .basic import LightGBMError, Predictor, Dataset, Booster, is_str
from . import callback



def _construct_dataset(x, y, reference=None,
    params=None, other_fields=None, predictor=None):
    if 'max_bin' in params:
        max_bin = int(params['max_bin'])
    else:
        max_bin = 255
    weight = None
    group = None
    init_score = None
    if other_fields is not None:
        if not is isinstance(other_fields, dict):
            raise TypeError("other filed data should be dict type")
        weight = None if 'weight' not in other_fields else other_fields['weight']
        group = None if 'group' not in other_fields else other_fields['group']
        init_score = None if 'init_score' not in other_fields else other_fields['init_score']
    if reference is None:
        ret = Dataset(x, y, max_bin=max_bin, 
            weight=weight, group=group, predictor=predictor, params=params)
    else:
        ret = reference.create_valid(x, y, weight, group, params=params)
    if init_score is not None:
        ret.set_init_score(init_score)
    return ret

def train(params, train_data, num_boost_round=100, 
        valid_datas=None, valid_names=None,
        fobj=None, feval=None, init_model=None, 
        train_fields=None, valid_fields=None, 
        early_stopping_rounds=None, out_eval_result=None,
        verbose_eval=True, learning_rates=None, callbacks=None):
    """Train with given parameters.

    Parameters
    ----------
    params : dict
         params.
    train_data : pair, (X, y)
        Data to be trained.
    num_boost_round: int
        Number of boosting iterations.
    valid_datas: list of pairs (valid_X, valid_y)
        List of data to be evaluated during training
    valid_names: list of string
        names of valid_datas
    fobj : function
        Customized objective function.
    feval : function
        Customized evaluation function.
        Note: should return (eval_name, eval_result, is_higher_better) of list of this
    init_model : file name of lightgbm model or 'Booster' instance
        model used for continued train
    train_fields : dict
        other data file in training data. e.g. train_fields['weight'] is weight data
        support fields: weight, group, init_score
    valid_fields : dict
        other data file in training data. e.g. valid_fields[0]['weight'] is weight data for first valid data
        support fields: weight, group, init_score
    early_stopping_rounds: int
        Activates early stopping. 
        Requires at least one validation data and one metric
        If there's more than one, will check all of them
        Returns the model with (best_iter + early_stopping_rounds)
        If early stopping occurs, the model will add 'best_iteration' field
    out_eval_result: dict or None
        This dictionary used to store all evaluation results of all the items in valid_datas.
        Example: with a valid_datas containing [dtest, dtrain] and valid_names containing ['eval', 'train'] and
        a paramater containing ('metric':'logloss')
        Returns: {'train': {'logloss': ['0.48253', '0.35953', ...]},
                  'eval': {'logloss': ['0.480385', '0.357756', ...]}}
        passed with None means no using this function
    verbose_eval : bool or int
        Requires at least one item in evals.
        If `verbose_eval` is True then the evaluation metric on the validation set is
        printed at each boosting stage.
        If `verbose_eval` is an integer then the evaluation metric on the validation set
        is printed at every given `verbose_eval` boosting stage. The last boosting stage
        / the boosting stage found by using `early_stopping_rounds` is also printed.
        Example: with verbose_eval=4 and at least one item in evals, an evaluation metric
        is printed every 4 boosting stages, instead of every boosting stage.
    learning_rates: list or function
        List of learning rate for each boosting round
        or a customized function that calculates learning_rate in terms of
        current number of round and the total number of boosting round (e.g. yields
        learning rate decay)
        - list l: learning_rate = l[current_round]
        - function f: learning_rate = f(current_round, total_boost_round)
    callbacks : list of callback functions
        List of callback functions that are applied at end of each iteration.

    Returns
    -------
    booster : a trained booster model
    """
    """create predictor first"""
    if is_str(init_model):
        predictor = Predictor(model_file=init_model)
    elif isinstance(init_model, Booster):
        predictor = Booster.to_predictor()
    elif isinstance(init_model, Predictor):
        predictor = init_model
    else:
        predictor = None
    """create dataset"""
    train_set = _construct_dataset(train_data[0], train_data[1], None, params, train_fields, predictor, silent)
    is_valid_contain_train = False
    train_data_name = "training"
    valid_sets = []
    name_valid_sets = []
    if valid_datas is not None:
        for i in range(len(valid_datas)):
            other_fields = None if valid_fields is None else valid_fields[i]
            """reduce cost for prediction training data"""
            if valid_datas[i] is train_data:
                is_valid_contain_train = True
                train_data_name = valid_names[i]
                continue
            valid_set = _construct_dataset(
                valid_datas[i][0], 
                valid_datas[i][1],
                train_set, 
                params, 
                other_fields, 
                predictor,
                silent)
            valid_sets.append(valid_set)
            name_valid_sets.append(valid_names[i])
    """process callbacks"""
    callbacks = [] if callbacks is None else callbacks

    # Most of legacy advanced options becomes callbacks
    if isinstance(verbose_eval, bool) and verbose_eval:
        callbacks.append(callback.print_evaluation())
    else:
        if isinstance(verbose_eval, int):
            callbacks.append(callback.print_evaluation(verbose_eval))

    if early_stopping_rounds is not None:
        callbacks.append(callback.early_stop(early_stopping_rounds,
                                             verbose=bool(verbose_eval)))
    if learning_rates is not None:
        callbacks.append(callback.reset_learning_rate(learning_rates))

    if evals_result is not None:
        callbacks.append(callback.record_evaluation(evals_result))

    callbacks_before_iter = [
        cb for cb in callbacks if cb.__dict__.get('before_iteration', False)]
    callbacks_after_iter = [
        cb for cb in callbacks if not cb.__dict__.get('before_iteration', False)]
    """construct booster"""
    booster = Booster(params=params, train_set=train_set, silent=silent)
    if is_valid_contain_train:
        booster.set_train_data_name(train_data_name)
    for i in range(len(valid_sets)):
        booster.add_valid(valid_sets[i], name_valid_sets[i])
    """start training"""
    for i in range(num_boost_round):
        for cb in callbacks_before_iter:
            cb(CallbackEnv(model=booster,
                           cvfolds=None,
                           iteration=i,
                           begin_iteration=0,
                           end_iteration=num_boost_round,
                           evaluation_result_list=None))

        booster.update(fobj=fobj)

        evaluation_result_list = []
        # check evaluation result.
        if len(valid_sets) != 0:
            if is_valid_contain_train:
                evaluation_result_list.extend(booster.eval_train(feval))
            evaluation_result_list.extend(booster.eval_valid(feval))
        try:
            for cb in callbacks_after_iter:
                cb(CallbackEnv(model=booster,
                               cvfolds=None,
                               iteration=i,
                               begin_iteration=0,
                               end_iteration=num_boost_round,
                               evaluation_result_list=evaluation_result_list))
        except EarlyStopException:
            break
    if booster.attr('best_iteration') is not None:
        booster.best_iteration = int(booster.attr('best_iteration'))
    else:
        booster.best_iteration = num_boost_round - 1
    return num_boost_round