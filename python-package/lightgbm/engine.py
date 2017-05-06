# coding: utf-8
# pylint: disable = invalid-name, W0105
"""Training Library containing training routines of LightGBM."""
from __future__ import absolute_import

import collections
from operator import attrgetter

import numpy as np

from . import callback
from .basic import Booster, Dataset, LightGBMError, _InnerPredictor
from .compat import (SKLEARN_INSTALLED, LGBMStratifiedKFold, LGBMGroupKFold, integer_types,
                     range_, string_type)


def train(params, train_set, num_boost_round=100,
          valid_sets=None, valid_names=None,
          fobj=None, feval=None, init_model=None,
          feature_name='auto', categorical_feature='auto',
          early_stopping_rounds=None, evals_result=None,
          verbose_eval=True, learning_rates=None, callbacks=None):
    """
    Train with given parameters.

    Parameters
    ----------
    params : dict
        Parameters for training.
    train_set : Dataset
        Data to be trained.
    num_boost_round: int
        Number of boosting iterations.
    valid_sets: list of Datasets
        List of data to be evaluated during training
    valid_names: list of string
        Names of valid_sets
    fobj : function
        Customized objective function.
    feval : function
        Customized evaluation function.
        Note: should return (eval_name, eval_result, is_higher_better) of list of this
    init_model : file name of lightgbm model or 'Booster' instance
        model used for continued train
    feature_name : list of str, or 'auto'
        Feature names
        If 'auto' and data is pandas DataFrame, use data columns name
    categorical_feature : list of str or int, or 'auto'
        Categorical features,
        type int represents index,
        type str represents feature names (need to specify feature_name as well)
        If 'auto' and data is pandas DataFrame, use pandas categorical columns
    early_stopping_rounds: int
        Activates early stopping.
        Requires at least one validation data and one metric
        If there's more than one, will check all of them
        Returns the model with (best_iter + early_stopping_rounds)
        If early stopping occurs, the model will add 'best_iteration' field
    evals_result: dict or None
        This dictionary used to store all evaluation results of all the items in valid_sets.
        Example: with a valid_sets containing [valid_set, train_set]
                 and valid_names containing ['eval', 'train']
                 and a paramater containing ('metric':'logloss')
        Returns: {'train': {'logloss': ['0.48253', '0.35953', ...]},
                  'eval': {'logloss': ['0.480385', '0.357756', ...]}}
        passed with None means no using this function
    verbose_eval : bool or int
        Requires at least one item in evals.
        If `verbose_eval` is True,
            the eval metric on the valid set is printed at each boosting stage.
        If `verbose_eval` is int,
            the eval metric on the valid set is printed at every `verbose_eval` boosting stage.
        The last boosting stage
            or the boosting stage found by using `early_stopping_rounds` is also printed.
        Example: with verbose_eval=4 and at least one item in evals,
            an evaluation metric is printed every 4 (instead of 1) boosting stages.
    learning_rates: list or function
        List of learning rate for each boosting round
        or a customized function that calculates learning_rate
        in terms of current number of round (e.g. yields learning rate decay)
        - list l: learning_rate = l[current_round]
        - function f: learning_rate = f(current_round)
    callbacks : list of callback functions
        List of callback functions that are applied at each iteration.
        See Callbacks in Python-API.md for more information.

    Returns
    -------
    booster : a trained booster model
    """
    """create predictor first"""
    if isinstance(init_model, string_type):
        predictor = _InnerPredictor(model_file=init_model)
    elif isinstance(init_model, Booster):
        predictor = init_model._to_predictor()
    else:
        predictor = None
    init_iteration = predictor.num_total_iteration if predictor is not None else 0
    """check dataset"""
    if not isinstance(train_set, Dataset):
        raise TypeError("Training only accepts Dataset object")

    train_set._update_params(params)
    train_set._set_predictor(predictor)
    train_set.set_feature_name(feature_name)
    train_set.set_categorical_feature(categorical_feature)

    is_valid_contain_train = False
    train_data_name = "training"
    reduced_valid_sets = []
    name_valid_sets = []
    if valid_sets is not None:
        if isinstance(valid_sets, Dataset):
            valid_sets = [valid_sets]
        if isinstance(valid_names, string_type):
            valid_names = [valid_names]
        for i, valid_data in enumerate(valid_sets):
            """reduce cost for prediction training data"""
            if valid_data is train_set:
                is_valid_contain_train = True
                if valid_names is not None:
                    train_data_name = valid_names[i]
                continue
            if not isinstance(valid_data, Dataset):
                raise TypeError("Traninig only accepts Dataset object")
            valid_data.set_reference(train_set)
            reduced_valid_sets.append(valid_data)
            if valid_names is not None and len(valid_names) > i:
                name_valid_sets.append(valid_names[i])
            else:
                name_valid_sets.append('valid_' + str(i))
        for valid_data in valid_sets:
            valid_data._update_params(params)
    """process callbacks"""
    if callbacks is None:
        callbacks = set()
    else:
        for i, cb in enumerate(callbacks):
            cb.__dict__.setdefault('order', i - len(callbacks))
        callbacks = set(callbacks)

    # Most of legacy advanced options becomes callbacks
    if verbose_eval is True:
        callbacks.add(callback.print_evaluation())
    elif isinstance(verbose_eval, integer_types):
        callbacks.add(callback.print_evaluation(verbose_eval))

    if early_stopping_rounds is not None:
        callbacks.add(callback.early_stopping(early_stopping_rounds, verbose=bool(verbose_eval)))

    if learning_rates is not None:
        callbacks.add(callback.reset_parameter(learning_rate=learning_rates))

    if evals_result is not None:
        callbacks.add(callback.record_evaluation(evals_result))

    callbacks_before_iter = {cb for cb in callbacks if getattr(cb, 'before_iteration', False)}
    callbacks_after_iter = callbacks - callbacks_before_iter
    callbacks_before_iter = sorted(callbacks_before_iter, key=attrgetter('order'))
    callbacks_after_iter = sorted(callbacks_after_iter, key=attrgetter('order'))

    """construct booster"""
    booster = Booster(params=params, train_set=train_set)
    if is_valid_contain_train:
        booster.set_train_data_name(train_data_name)
    for valid_set, name_valid_set in zip(reduced_valid_sets, name_valid_sets):
        booster.add_valid(valid_set, name_valid_set)
    booster.best_iteration = 0

    """start training"""
    for i in range_(init_iteration, init_iteration + num_boost_round):
        for cb in callbacks_before_iter:
            cb(callback.CallbackEnv(model=booster,
                                    params=params,
                                    iteration=i,
                                    begin_iteration=init_iteration,
                                    end_iteration=init_iteration + num_boost_round,
                                    evaluation_result_list=None))

        booster.update(fobj=fobj)

        evaluation_result_list = []
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
    booster.best_score = collections.defaultdict(dict)
    for dataset_name, eval_name, score, _ in evaluation_result_list:
        booster.best_score[dataset_name][eval_name] = score
    return booster


class CVBooster(object):
    """"Auxiliary data struct to hold all boosters of CV."""
    def __init__(self):
        self.boosters = []
        self.best_iteration = -1

    def append(self, booster):
        """add a booster to CVBooster"""
        self.boosters.append(booster)

    def __getattr__(self, name):
        """redirect methods call of CVBooster"""
        def handlerFunction(*args, **kwargs):
            """call methods with each booster, and concatenate their results"""
            ret = []
            for booster in self.boosters:
                ret.append(getattr(booster, name)(*args, **kwargs))
            return ret
        return handlerFunction


def _make_n_folds(full_data, folds, nfold, params, seed, fpreproc=None, stratified=False, shuffle=True):
    """
    Make an n-fold list of Booster from random indices.
    """
    full_data = full_data.construct()
    num_data = full_data.num_data()
    if folds is not None:
        if not hasattr(folds, '__iter__'):
            raise AttributeError("folds should be a generator or iterator of (train_idx, test_idx)")
    else:
        if 'objective' in params and params['objective'] == 'lambdarank':
            if not SKLEARN_INSTALLED:
                raise LightGBMError('Scikit-learn is required for lambdarank cv.')
            # lambdarank task, split according to groups
            group_info = full_data.get_group().astype(int)
            flatted_group = np.repeat(range(len(group_info)), repeats=group_info)
            group_kfold = LGBMGroupKFold(n_splits=nfold)
            folds = group_kfold.split(X=np.zeros(num_data), groups=flatted_group)
        elif stratified:
            if not SKLEARN_INSTALLED:
                raise LightGBMError('Scikit-learn is required for stratified cv.')
            skf = LGBMStratifiedKFold(n_splits=nfold, shuffle=shuffle, random_state=seed)
            folds = skf.split(X=np.zeros(num_data), y=full_data.get_label())
        else:
            if shuffle:
                randidx = np.random.RandomState(seed).permutation(num_data)
            else:
                randidx = np.arange(num_data)
            kstep = int(num_data / nfold)
            test_id = [randidx[i: i + kstep] for i in range_(0, num_data, kstep)]
            train_id = [np.concatenate([test_id[i] for i in range_(nfold) if k != i]) for k in range_(nfold)]
            folds = zip(train_id, test_id)

    ret = CVBooster()
    for train_idx, test_idx in folds:
        train_set = full_data.subset(train_idx)
        valid_set = full_data.subset(test_idx)
        # run preprocessing on the data set if needed
        if fpreproc is not None:
            train_set, valid_set, tparam = fpreproc(train_set, valid_set, params.copy())
        else:
            tparam = params
        cvbooster = Booster(tparam, train_set)
        cvbooster.add_valid(valid_set, 'valid')
        ret.append(cvbooster)
    return ret


def _agg_cv_result(raw_results):
    """
    Aggregate cross-validation results.
    """
    cvmap = collections.defaultdict(list)
    metric_type = {}
    for one_result in raw_results:
        for one_line in one_result:
            metric_type[one_line[1]] = one_line[3]
            cvmap[one_line[1]].append(one_line[2])
    return [('cv_agg', k, np.mean(v), metric_type[k], np.std(v)) for k, v in cvmap.items()]


def cv(params, train_set, num_boost_round=10,
       folds=None, nfold=5, stratified=False, shuffle=True,
       metrics=None, fobj=None, feval=None, init_model=None,
       feature_name='auto', categorical_feature='auto',
       early_stopping_rounds=None, fpreproc=None,
       verbose_eval=None, show_stdv=True, seed=0,
       callbacks=None):
    """
    Cross-validation with given paramaters.

    Parameters
    ----------
    params : dict
        Booster params.
    train_set : Dataset
        Data to be trained.
    num_boost_round : int
        Number of boosting iterations.
    folds : a generator or iterator of (train_idx, test_idx) tuples
        The train indices and test indices for each folds.
        This argument has highest priority over other data split arguments.
    nfold : int
        Number of folds in CV.
    stratified : bool
        Perform stratified sampling.
    shuffle: bool
        Whether shuffle before split data
    metrics : string or list of strings
        Evaluation metrics to be watched in CV.
        If `metrics` is not None, the metric in `params` will be overridden.
    fobj : function
        Custom objective function.
    feval : function
        Custom evaluation function.
    init_model : file name of lightgbm model or 'Booster' instance
        model used for continued train
    feature_name : list of str, or 'auto'
        Feature names
        If 'auto' and data is pandas DataFrame, use data columns name
    categorical_feature : list of str or int, or 'auto'
        Categorical features,
        type int represents index,
        type str represents feature names (need to specify feature_name as well)
        If 'auto' and data is pandas DataFrame, use pandas categorical columns
    early_stopping_rounds: int
        Activates early stopping. CV error needs to decrease at least
        every <early_stopping_rounds> round(s) to continue.
        Last entry in evaluation history is the one from best iteration.
    fpreproc : function
        Preprocessing function that takes (dtrain, dtest, param)
        and returns transformed versions of those.
    verbose_eval : bool, int, or None, default None
        Whether to display the progress.
        If None, progress will be displayed when np.ndarray is returned.
        If True, progress will be displayed at boosting stage.
        If an integer is given,
            progress will be displayed at every given `verbose_eval` boosting stage.
    show_stdv : bool, default True
        Whether to display the standard deviation in progress.
        Results are not affected, and always contains std.
    seed : int
        Seed used to generate the folds (passed to numpy.random.seed).
    callbacks : list of callback functions
        List of callback functions that are applied at each iteration.
        See Callbacks in Python-API.md for more information.

    Returns
    -------
    evaluation history : list(string)
    """
    if not isinstance(train_set, Dataset):
        raise TypeError("Traninig only accepts Dataset object")

    if isinstance(init_model, string_type):
        predictor = _InnerPredictor(model_file=init_model)
    elif isinstance(init_model, Booster):
        predictor = init_model._to_predictor()
    else:
        predictor = None
    train_set._update_params(params)
    train_set._set_predictor(predictor)
    train_set.set_feature_name(feature_name)
    train_set.set_categorical_feature(categorical_feature)

    if metrics is not None:
        params['metric'] = metrics

    results = collections.defaultdict(list)
    cvfolds = _make_n_folds(train_set, folds=folds, nfold=nfold,
                            params=params, seed=seed, fpreproc=fpreproc,
                            stratified=stratified, shuffle=shuffle)

    # setup callbacks
    if callbacks is None:
        callbacks = set()
    else:
        for i, cb in enumerate(callbacks):
            cb.__dict__.setdefault('order', i - len(callbacks))
        callbacks = set(callbacks)
    if early_stopping_rounds is not None:
        callbacks.add(callback.early_stopping(early_stopping_rounds, verbose=False))
    if verbose_eval is True:
        callbacks.add(callback.print_evaluation(show_stdv=show_stdv))
    elif isinstance(verbose_eval, integer_types):
        callbacks.add(callback.print_evaluation(verbose_eval, show_stdv=show_stdv))

    callbacks_before_iter = {cb for cb in callbacks if getattr(cb, 'before_iteration', False)}
    callbacks_after_iter = callbacks - callbacks_before_iter
    callbacks_before_iter = sorted(callbacks_before_iter, key=attrgetter('order'))
    callbacks_after_iter = sorted(callbacks_after_iter, key=attrgetter('order'))

    for i in range_(num_boost_round):
        for cb in callbacks_before_iter:
            cb(callback.CallbackEnv(model=cvfolds,
                                    params=params,
                                    iteration=i,
                                    begin_iteration=0,
                                    end_iteration=num_boost_round,
                                    evaluation_result_list=None))
        cvfolds.update(fobj=fobj)
        res = _agg_cv_result(cvfolds.eval_valid(feval))
        for _, key, mean, _, std in res:
            results[key + '-mean'].append(mean)
            results[key + '-stdv'].append(std)
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
            for k in results:
                results[k] = results[k][:cvfolds.best_iteration]
            break
    return dict(results)
