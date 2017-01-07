# coding: utf-8
# pylint: disable = invalid-name, W0105, C0111, C0301
"""Scikit-Learn Wrapper interface for LightGBM."""
from __future__ import absolute_import

import inspect

import numpy as np

from .basic import IS_PY3, Dataset, LightGBMError
from .engine import train

'''sklearn'''
try:
    from sklearn.base import BaseEstimator
    from sklearn.base import RegressorMixin, ClassifierMixin
    from sklearn.preprocessing import LabelEncoder
    from sklearn.utils import deprecated
    SKLEARN_INSTALLED = True
    LGBMModelBase = BaseEstimator
    LGBMRegressorBase = RegressorMixin
    LGBMClassifierBase = ClassifierMixin
    LGBMLabelEncoder = LabelEncoder
except ImportError:
    SKLEARN_INSTALLED = False
    LGBMModelBase = object
    LGBMClassifierBase = object
    LGBMRegressorBase = object
    LGBMLabelEncoder = None


def _argc(func):
    if IS_PY3:
        return len(inspect.signature(func).parameters)
    else:
        return len(inspect.getargspec(func).args)


def _objective_function_wrapper(func):
    """Decorate an objective function
    Note: for multi-class task, the y_pred is group by class_id first, then group by row_id
          if you want to get i-th row y_pred in j-th class, the access way is y_pred[j*num_data+i]
          and you should group grad and hess in this way as well
    Parameters
    ----------
    func: callable
        Expects a callable with signature ``func(y_true, y_pred)`` or ``func(y_true, y_pred, group):
            y_true: array_like of shape [n_samples]
                The target values
            y_pred: array_like of shape [n_samples] or shape[n_samples * n_class] (for multi-class)
                The predicted values
            group: array_like
                group/query data, used for ranking task

    Returns
    -------
    new_func: callable
        The new objective function as expected by ``lightgbm.engine.train``.
        The signature is ``new_func(preds, dataset)``:

        preds: array_like, shape [n_samples] or shape[n_samples * n_class]
            The predicted values
        dataset: ``dataset``
            The training set from which the labels will be extracted using
            ``dataset.get_label()``
    """
    def inner(preds, dataset):
        """internal function"""
        labels = dataset.get_label()
        argc = _argc(func)
        if argc == 2:
            grad, hess = func(labels, preds)
        elif argc == 3:
            grad, hess = func(labels, preds, dataset.get_group())
        else:
            raise TypeError("Self-defined objective function should have 2 or 3 arguments, got %d" % argc)
        """weighted for objective"""
        weight = dataset.get_weight()
        if weight is not None:
            """only one class"""
            if len(weight) == len(grad):
                grad = np.multiply(grad, weight)
                hess = np.multiply(hess, weight)
            else:
                num_data = len(weight)
                num_class = len(grad) // num_data
                if num_class * num_data != len(grad):
                    raise ValueError("Length of grad and hess should equal to num_class * num_data")
                for k in range(num_class):
                    for i in range(num_data):
                        idx = k * num_data + i
                        grad[idx] *= weight[i]
                        hess[idx] *= weight[i]
        return grad, hess
    return inner


def _eval_function_wrapper(func):
    """Decorate an eval function
    Note: for multi-class task, the y_pred is group by class_id first, then group by row_id
          if you want to get i-th row y_pred in j-th class, the access way is y_pred[j*num_data+i]
    Parameters
    ----------
    func: callable
        Expects a callable with following functions:
            ``func(y_true, y_pred)``,
            ``func(y_true, y_pred, weight)``
         or ``func(y_true, y_pred, weight, group)``
            and return (eval_name->str, eval_result->float, is_bigger_better->Bool):

            y_true: array_like of shape [n_samples]
                The target values
            y_pred: array_like of shape [n_samples] or shape[n_samples * n_class] (for multi-class)
                The predicted values
            weight: array_like of shape [n_samples]
                The weight of samples
            group: array_like
                group/query data, used for ranking task

    Returns
    -------
    new_func: callable
        The new eval function as expected by ``lightgbm.engine.train``.
        The signature is ``new_func(preds, dataset)``:

        preds: array_like, shape [n_samples] or shape[n_samples * n_class]
            The predicted values
        dataset: ``dataset``
            The training set from which the labels will be extracted using
            ``dataset.get_label()``
    """
    def inner(preds, dataset):
        """internal function"""
        labels = dataset.get_label()
        argc = _argc(func)
        if argc == 2:
            return func(labels, preds)
        elif argc == 3:
            return func(labels, preds, dataset.get_weight())
        elif argc == 4:
            return func(labels, preds, dataset.get_weight(), dataset.get_group())
        else:
            raise TypeError("Self-defined eval function should have 2, 3 or 4 arguments, got %d" % argc)
    return inner


class LGBMModel(LGBMModelBase):

    def __init__(self, boosting_type="gbdt", num_leaves=31, max_depth=-1,
                 learning_rate=0.1, n_estimators=10, max_bin=255,
                 subsample_for_bin=50000, objective="regression",
                 min_split_gain=0, min_child_weight=5, min_child_samples=10,
                 subsample=1, subsample_freq=1, colsample_bytree=1,
                 reg_alpha=0, reg_lambda=0, scale_pos_weight=1,
                 is_unbalance=False, seed=0, nthread=-1, silent=True,
                 sigmoid=1.0, max_position=20, label_gain=None,
                 drop_rate=0.1, skip_drop=0.5, max_drop=50,
                 uniform_drop=False, xgboost_dart_mode=False):
        """
        Implementation of the Scikit-Learn API for LightGBM.

        Parameters
        ----------
        boosting_type : string
            gbdt, traditional Gradient Boosting Decision Tree
            dart, Dropouts meet Multiple Additive Regression Trees
        num_leaves : int
            Maximum tree leaves for base learners.
        max_depth : int
            Maximum tree depth for base learners, -1 means no limit.
        learning_rate : float
            Boosting learning rate
        n_estimators : int
            Number of boosted trees to fit.
        max_bin : int
            Number of bucketed bin for feature values
        subsample_for_bin : int
            Number of samples for constructing bins.
        objective : string or callable
            Specify the learning task and the corresponding learning objective or
            a custom objective function to be used (see note below).
            default: binary for LGBMClassifier, lambdarank for LGBMRanker
        min_split_gain : float
            Minimum loss reduction required to make a further partition on a leaf node of the tree.
        min_child_weight : int
            Minimum sum of instance weight(hessian) needed in a child(leaf)
        min_child_samples : int
            Minimum number of data need in a child(leaf)
        subsample : float
            Subsample ratio of the training instance.
        subsample_freq : int
            frequence of subsample, <=0 means no enable
        colsample_bytree : float
            Subsample ratio of columns when constructing each tree.
        reg_alpha : float
            L1 regularization term on weights
        reg_lambda : float
            L2 regularization term on weights
        scale_pos_weight : float
            Balancing of positive and negative weights.
        is_unbalance : bool
            Is unbalance for binary classification
        seed : int
            Random number seed.
        nthread : int
            Number of parallel threads
        silent : boolean
            Whether to print messages while running boosting.
        sigmoid : float
            Only used in binary classification and lambdarank. Parameter for sigmoid function.
        max_position : int
            Only used in lambdarank, will optimize NDCG at this position.
        label_gain : list of float
            Only used in lambdarank, relevant gain for labels.
            For example, the gain of label 2 is 3 if using default label gains.
            None (default) means use default value of CLI version: {0,1,3,7,15,31,63,...}.
        drop_rate : float
            Only used when boosting_type='dart'. Probablity to select dropping trees.
        skip_drop : float
            Only used when boosting_type='dart'. Probablity to skip dropping trees.
        max_drop : int
            Only used when boosting_type='dart'. Max number of dropped trees in one iteration.
        uniform_drop : bool
            Only used when boosting_type='dart'. If true, drop trees uniformly, else drop according to weights.
        xgboost_dart_mode : bool
            Only used when boosting_type='dart'. Whether use xgboost dart mode.

        Note
        ----
        A custom objective function can be provided for the ``objective``
        parameter. In this case, it should have the signature
        ``objective(y_true, y_pred) -> grad, hess``
            or ``objective(y_true, y_pred, group) -> grad, hess``:

            y_true: array_like of shape [n_samples]
                The target values
            y_pred: array_like of shape [n_samples] or shape[n_samples * n_class]
                The predicted values
            group: array_like
                group/query data, used for ranking task
            grad: array_like of shape [n_samples] or shape[n_samples * n_class]
                The value of the gradient for each sample point.
            hess: array_like of shape [n_samples] or shape[n_samples * n_class]
                The value of the second derivative for each sample point

        for multi-class task, the y_pred is group by class_id first, then group by row_id
            if you want to get i-th row y_pred in j-th class, the access way is y_pred[j*num_data+i]
            and you should group grad and hess in this way as well
        """
        if not SKLEARN_INSTALLED:
            raise LightGBMError('Scikit-learn is required for this module')

        self.boosting_type = boosting_type
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_bin = max_bin
        self.subsample_for_bin = subsample_for_bin
        self.objective = objective
        self.min_split_gain = min_split_gain
        self.min_child_weight = min_child_weight
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.subsample_freq = subsample_freq
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        self.is_unbalance = is_unbalance
        self.seed = seed
        self.nthread = nthread
        self.silent = silent
        self.sigmoid = sigmoid
        self.max_position = max_position
        self.label_gain = label_gain
        self.drop_rate = drop_rate
        self.skip_drop = skip_drop
        self.max_drop = max_drop
        self.uniform_drop = uniform_drop
        self.xgboost_dart_mode = xgboost_dart_mode
        self._Booster = None
        self.evals_result = None
        self.best_iteration = -1
        if callable(self.objective):
            self.fobj = _objective_function_wrapper(self.objective)
        else:
            self.fobj = None

    def fit(self, X, y,
            sample_weight=None, init_score=None, group=None,
            eval_set=None, eval_sample_weight=None,
            eval_init_score=None, eval_group=None,
            eval_metric=None,
            early_stopping_rounds=None, verbose=True,
            feature_name=None, categorical_feature=None,
            callbacks=None):
        """
        Fit the gradient boosting model

        Parameters
        ----------
        X : array_like
            Feature matrix
        y : array_like
            Labels
        sample_weight : array_like
            weight of training data
        init_score : array_like
            init score of training data
        group : array_like
            group data of training data
        eval_set : list, optional
            A list of (X, y) tuple pairs to use as a validation set for early-stopping
        eval_sample_weight : List of array
            weight of eval data
        eval_init_score : List of array
            init score of eval data
        eval_group : List of array
            group data of eval data
        eval_metric : str, list of str, callable, optional
            If a str, should be a built-in evaluation metric to use.
            If callable, a custom evaluation metric, see note for more details.
        early_stopping_rounds : int
        verbose : bool
            If `verbose` and an evaluation set is used, writes the evaluation
        feature_name : list of str
            Feature names
        categorical_feature : list of str or int
            Categorical features,
            type int represents index,
            type str represents feature names (need to specify feature_name as well)
        callbacks : list of callback functions
            List of callback functions that are applied at each iteration.
            See Callbacks in Python-API.md for more information.

        Note
        ----
        Custom eval function expects a callable with following functions:
            ``func(y_true, y_pred)``, ``func(y_true, y_pred, weight)``
                or ``func(y_true, y_pred, weight, group)``.
            return (eval_name, eval_result, is_bigger_better)
                or list of (eval_name, eval_result, is_bigger_better)

            y_true: array_like of shape [n_samples]
                The target values
            y_pred: array_like of shape [n_samples] or shape[n_samples * n_class] (for multi-class)
                The predicted values
            weight: array_like of shape [n_samples]
                The weight of samples
            group: array_like
                group/query data, used for ranking task
            eval_name: str
                name of evaluation
            eval_result: float
                eval result
            is_bigger_better: bool
                is eval result bigger better, e.g. AUC is bigger_better.
        for multi-class task, the y_pred is group by class_id first, then group by row_id
          if you want to get i-th row y_pred in j-th class, the access way is y_pred[j*num_data+i]
        """
        evals_result = {}
        params = self.get_params()
        params['verbose'] = -1 if self.silent else 1
        if hasattr(self, 'n_classes_') and self.n_classes_ > 2:
            params['num_class'] = self.n_classes_
        if hasattr(self, 'eval_at'):
            params['ndcg_eval_at'] = self.eval_at
        if self.fobj:
            params['objective'] = 'None'  # objective = nullptr for unknown objective
        if 'label_gain' in params and params['label_gain'] is None:
            del params['label_gain']  # use default of cli version

        if callable(eval_metric):
            feval = _eval_function_wrapper(eval_metric)
        else:
            feval = None
            params['metric'] = eval_metric

        def _construct_dataset(X, y, sample_weight, init_score, group, params):
            ret = Dataset(X, label=y, max_bin=self.max_bin, weight=sample_weight, group=group, params=params)
            ret.set_init_score(init_score)
            return ret

        train_set = _construct_dataset(X, y, sample_weight, init_score, group, params)

        valid_sets = []
        if eval_set is not None:
            if isinstance(eval_set, tuple):
                eval_set = [eval_set]
            for i, valid_data in enumerate(eval_set):
                """reduce cost for prediction training data"""
                if valid_data[0] is X and valid_data[1] is y:
                    valid_set = train_set
                else:
                    def get_meta_data(collection, i):
                        if collection is None:
                            return None
                        elif isinstance(collection, list):
                            return collection[i] if len(collection) > i else None
                        elif isinstance(collection, dict):
                            return collection.get(i, None)
                        else:
                            raise TypeError('eval_sample_weight, eval_init_score, and eval_group should be dict or list')
                    valid_weight = get_meta_data(eval_sample_weight, i)
                    valid_init_score = get_meta_data(eval_init_score, i)
                    valid_group = get_meta_data(eval_group, i)
                    valid_set = _construct_dataset(valid_data[0], valid_data[1], valid_weight, valid_init_score, valid_group, params)
                valid_sets.append(valid_set)

        self._Booster = train(params, train_set,
                              self.n_estimators, valid_sets=valid_sets,
                              early_stopping_rounds=early_stopping_rounds,
                              evals_result=evals_result, fobj=self.fobj, feval=feval,
                              verbose_eval=verbose, feature_name=feature_name,
                              categorical_feature=categorical_feature,
                              callbacks=callbacks)

        if evals_result:
            self.evals_result = evals_result

        if early_stopping_rounds is not None:
            self.best_iteration = self._Booster.best_iteration
        return self

    def predict(self, X, raw_score=False, num_iteration=0):
        """
        Return the predicted value for each sample.

        Parameters
        ----------
        X : array_like, shape=[n_samples, n_features]
            Input features matrix.

        num_iteration : int
            Limit number of iterations in the prediction; defaults to 0 (use all trees).

        Returns
        -------
        predicted_result : array_like, shape=[n_samples] or [n_samples, n_classes]
        """
        return self.booster_.predict(X, raw_score=raw_score, num_iteration=num_iteration)

    def apply(self, X, num_iteration=0):
        """
        Return the predicted leaf every tree for each sample.

        Parameters
        ----------
        X : array_like, shape=[n_samples, n_features]
            Input features matrix.

        num_iteration : int
            Limit number of iterations in the prediction; defaults to 0 (use all trees).

        Returns
        -------
        X_leaves : array_like, shape=[n_samples, n_trees]
        """
        return self.booster_.predict(X, pred_leaf=True, num_iteration=num_iteration)

    @property
    def booster_(self):
        """Get the underlying lightgbm Booster of this model."""
        if self._Booster is None:
            raise LightGBMError('No booster found. Need to call fit beforehand.')
        return self._Booster

    @property
    def evals_result_(self):
        """Get the evaluation results."""
        if self.evals_result is None:
            raise LightGBMError('No results found. Need to call fit with eval set beforehand.')
        return self.evals_result

    @property
    def feature_importance_(self):
        """Get normailized feature importances."""
        importace_array = self.booster_.feature_importance().astype(np.float32)
        return importace_array / importace_array.sum()

    @deprecated('Use attribute booster_ instead.')
    def booster(self):
        return self.booster_

    @deprecated('Use attribute feature_importance_ instead.')
    def feature_importance(self):
        return self.feature_importance_


class LGBMRegressor(LGBMModel, LGBMRegressorBase):

    def fit(self, X, y,
            sample_weight=None, init_score=None,
            eval_set=None, eval_sample_weight=None,
            eval_init_score=None,
            eval_metric="l2",
            early_stopping_rounds=None, verbose=True,
            feature_name=None, categorical_feature=None, callbacks=None):

        super(LGBMRegressor, self).fit(X, y, sample_weight=sample_weight,
                                       init_score=init_score, eval_set=eval_set,
                                       eval_sample_weight=eval_sample_weight,
                                       eval_init_score=eval_init_score,
                                       eval_metric=eval_metric,
                                       early_stopping_rounds=early_stopping_rounds,
                                       verbose=verbose, feature_name=feature_name,
                                       categorical_feature=categorical_feature,
                                       callbacks=callbacks)
        return self


class LGBMClassifier(LGBMModel, LGBMClassifierBase):

    def __init__(self, boosting_type="gbdt", num_leaves=31, max_depth=-1,
                 learning_rate=0.1, n_estimators=10, max_bin=255,
                 subsample_for_bin=50000, objective="binary",
                 min_split_gain=0, min_child_weight=5, min_child_samples=10,
                 subsample=1, subsample_freq=1, colsample_bytree=1,
                 reg_alpha=0, reg_lambda=0, scale_pos_weight=1,
                 is_unbalance=False, seed=0, nthread=-1,
                 silent=True, sigmoid=1.0,
                 drop_rate=0.1, skip_drop=0.5, max_drop=50,
                 uniform_drop=False, xgboost_dart_mode=False):
        self.classes, self.n_classes = None, None
        super(LGBMClassifier, self).__init__(boosting_type=boosting_type, num_leaves=num_leaves,
                                             max_depth=max_depth, learning_rate=learning_rate,
                                             n_estimators=n_estimators, max_bin=max_bin,
                                             subsample_for_bin=subsample_for_bin, objective=objective,
                                             min_split_gain=min_split_gain, min_child_weight=min_child_weight,
                                             min_child_samples=min_child_samples, subsample=subsample,
                                             subsample_freq=subsample_freq, colsample_bytree=colsample_bytree,
                                             reg_alpha=reg_alpha, reg_lambda=reg_lambda,
                                             scale_pos_weight=scale_pos_weight, is_unbalance=is_unbalance,
                                             seed=seed, nthread=nthread, silent=silent, sigmoid=sigmoid,
                                             drop_rate=drop_rate, skip_drop=skip_drop, max_drop=max_drop,
                                             uniform_drop=uniform_drop, xgboost_dart_mode=xgboost_dart_mode)

    def fit(self, X, y,
            sample_weight=None, init_score=None,
            eval_set=None, eval_sample_weight=None,
            eval_init_score=None,
            eval_metric="binary_logloss",
            early_stopping_rounds=None, verbose=True,
            feature_name=None, categorical_feature=None,
            callbacks=None):
        self._le = LGBMLabelEncoder().fit(y)
        y = self._le.transform(y)

        self.classes = self._le.classes_
        self.n_classes = len(self.classes_)
        if self.n_classes > 2:
            # Switch to using a multiclass objective in the underlying LGBM instance
            self.objective = "multiclass"
            if eval_set is not None and eval_metric == "binary_logloss":
                eval_metric = "multi_logloss"

        if eval_set is not None:
            eval_set = [(x[0], self._le.transform(x[1])) for x in eval_set]

        super(LGBMClassifier, self).fit(X, y, sample_weight=sample_weight,
                                        init_score=init_score, eval_set=eval_set,
                                        eval_sample_weight=eval_sample_weight,
                                        eval_init_score=eval_init_score,
                                        eval_metric=eval_metric,
                                        early_stopping_rounds=early_stopping_rounds,
                                        verbose=verbose, feature_name=feature_name,
                                        categorical_feature=categorical_feature,
                                        callbacks=callbacks)
        return self

    def predict(self, X, raw_score=False, num_iteration=0):
        class_probs = self.predict_proba(X, raw_score, num_iteration)
        class_index = np.argmax(class_probs, axis=1)
        return self._le.inverse_transform(class_index)

    def predict_proba(self, X, raw_score=False, num_iteration=0):
        """
        Return the predicted probability for each class for each sample.

        Parameters
        ----------
        X : array_like, shape=[n_samples, n_features]
            Input features matrix.

        num_iteration : int
            Limit number of iterations in the prediction; defaults to 0 (use all trees).

        Returns
        -------
        predicted_probability : array_like, shape=[n_samples, n_classes]
        """
        class_probs = self.booster_.predict(X, raw_score=raw_score, num_iteration=num_iteration)
        if self.n_classes > 2:
            return class_probs
        else:
            return np.vstack((1. - class_probs, class_probs)).transpose()

    @property
    def classes_(self):
        """Get class label array."""
        if self.classes is None:
            raise LightGBMError('No classes found. Need to call fit beforehand.')
        return self.classes

    @property
    def n_classes_(self):
        """Get number of classes"""
        if self.n_classes is None:
            raise LightGBMError('No classes found. Need to call fit beforehand.')
        return self.n_classes


class LGBMRanker(LGBMModel):

    def __init__(self, boosting_type="gbdt", num_leaves=31, max_depth=-1,
                 learning_rate=0.1, n_estimators=10, max_bin=255,
                 subsample_for_bin=50000, objective="lambdarank",
                 min_split_gain=0, min_child_weight=5, min_child_samples=10,
                 subsample=1, subsample_freq=1, colsample_bytree=1,
                 reg_alpha=0, reg_lambda=0, scale_pos_weight=1,
                 is_unbalance=False, seed=0, nthread=-1, silent=True,
                 sigmoid=1.0, max_position=20, label_gain=None,
                 drop_rate=0.1, skip_drop=0.5, max_drop=50,
                 uniform_drop=False, xgboost_dart_mode=False):
        super(LGBMRanker, self).__init__(boosting_type=boosting_type, num_leaves=num_leaves,
                                         max_depth=max_depth, learning_rate=learning_rate,
                                         n_estimators=n_estimators, max_bin=max_bin,
                                         subsample_for_bin=subsample_for_bin, objective=objective,
                                         min_split_gain=min_split_gain, min_child_weight=min_child_weight,
                                         min_child_samples=min_child_samples, subsample=subsample,
                                         subsample_freq=subsample_freq, colsample_bytree=colsample_bytree,
                                         reg_alpha=reg_alpha, reg_lambda=reg_lambda,
                                         scale_pos_weight=scale_pos_weight, is_unbalance=is_unbalance,
                                         seed=seed, nthread=nthread, silent=silent,
                                         sigmoid=sigmoid, max_position=max_position, label_gain=label_gain,
                                         drop_rate=drop_rate, skip_drop=skip_drop, max_drop=max_drop,
                                         uniform_drop=uniform_drop, xgboost_dart_mode=xgboost_dart_mode)

    def fit(self, X, y,
            sample_weight=None, init_score=None, group=None,
            eval_set=None, eval_sample_weight=None,
            eval_init_score=None, eval_group=None,
            eval_metric='ndcg', eval_at=1,
            early_stopping_rounds=None, verbose=True,
            feature_name=None, categorical_feature=None,
            callbacks=None):
        """
        Most arguments like common methods except following:

        eval_at : list of int
            The evaulation positions of NDCG
        """

        """check group data"""
        if group is None:
            raise ValueError("Should set group for ranking task")

        if eval_set is not None:
            if eval_group is None:
                raise ValueError("Eval_group cannot be None when eval_set is not None")
            elif len(eval_group) != len(eval_set):
                raise ValueError("Length of eval_group should equal to eval_set")
            elif (isinstance(eval_group, dict) and any(i not in eval_group or eval_group[i] is None for i in range(len(eval_group)))) \
                    or (isinstance(eval_group, list) and any(group is None for group in eval_group)):
                raise ValueError("Should set group for all eval dataset for ranking task; if you use dict, the index should start from 0")

        if eval_at is not None:
            self.eval_at = eval_at
        super(LGBMRanker, self).fit(X, y, sample_weight=sample_weight,
                                    init_score=init_score, group=group,
                                    eval_set=eval_set, eval_sample_weight=eval_sample_weight,
                                    eval_init_score=eval_init_score, eval_group=eval_group,
                                    eval_metric=eval_metric,
                                    early_stopping_rounds=early_stopping_rounds,
                                    verbose=verbose, feature_name=feature_name,
                                    categorical_feature=categorical_feature,
                                    callbacks=callbacks)
        return self
