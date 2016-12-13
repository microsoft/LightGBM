# coding: utf-8
# pylint: disable = invalid-name, W0105, C0111, C0301
"""Scikit-Learn Wrapper interface for LightGBM."""
from __future__ import absolute_import
import inspect

import numpy as np
from .basic import LightGBMError, Dataset, is_str
from .engine import train
'''sklearn'''
try:
    from sklearn.base import BaseEstimator
    from sklearn.base import RegressorMixin, ClassifierMixin
    from sklearn.preprocessing import LabelEncoder
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
            y_pred: array_like of shape [n_samples] or shape[n_samples* n_class] (for multi-class)
                The predicted values
            group: array_like
                group/query data, used for ranking task

    Returns
    -------
    new_func: callable
        The new objective function as expected by ``lightgbm.engine.train``.
        The signature is ``new_func(preds, dataset)``:

        preds: array_like, shape [n_samples] or shape[n_samples* n_class]
            The predicted values
        dataset: ``dataset``
            The training set from which the labels will be extracted using
            ``dataset.get_label()``
    """
    def inner(preds, dataset):
        """internal function"""
        labels = dataset.get_label()
        argc = len(inspect.getargspec(func).args)
        if argc == 2:
            grad, hess = func(labels, preds)
        elif argc == 3:
            grad, hess = func(labels, preds, dataset.get_group())
        else:
            raise TypeError("Self-defined objective function should have 2 or 3 arguments, got %d" %(argc))
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
            y_pred: array_like of shape [n_samples] or shape[n_samples* n_class] (for multi-class)
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

        preds: array_like, shape [n_samples] or shape[n_samples* n_class]
            The predicted values
        dataset: ``dataset``
            The training set from which the labels will be extracted using
            ``dataset.get_label()``
    """
    def inner(preds, dataset):
        """internal function"""
        labels = dataset.get_label()
        argc = len(inspect.getargspec(func).args)
        if argc == 2:
            return func(labels, preds)
        elif argc == 3:
            return func(labels, preds, dataset.get_weight())
        elif argc == 4:
            return func(labels, preds, dataset.get_weight(), dataset.get_group())
        else:
            raise TypeError("Self-defined eval function should have 2, 3 or 4 arguments, got %d" %(argc))
    return inner

class LGBMModel(LGBMModelBase):

    def __init__(self, boosting_type="gbdt", num_leaves=31, max_depth=-1,
                 learning_rate=0.1, n_estimators=10, max_bin=255,
                 silent=True, objective="regression",
                 nthread=-1, min_split_gain=0, min_child_weight=5, min_child_samples=10,
                 subsample=1, subsample_freq=1, colsample_bytree=1,
                 reg_alpha=0, reg_lambda=0, scale_pos_weight=1,
                 is_unbalance=False, seed=0):
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
        silent : boolean
            Whether to print messages while running boosting.
        objective : string or callable
            Specify the learning task and the corresponding learning objective or
            a custom objective function to be used (see note below).
            default: binary for LGBMClassifier, lambdarank for LGBMRanker
        nthread : int
            Number of parallel threads
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

        Note
        ----
        A custom objective function can be provided for the ``objective``
        parameter. In this case, it should have the signature
        ``objective(y_true, y_pred) -> grad, hess``
            or ``objective(y_true, y_pred, group) -> grad, hess``:

            y_true: array_like of shape [n_samples]
                The target values
            y_pred: array_like of shape [n_samples] or shape[n_samples* n_class]
                The predicted values
            group: array_like
                group/query data, used for ranking task
            grad: array_like of shape [n_samples] or shape[n_samples* n_class]
                The value of the gradient for each sample point.
            hess: array_like of shape [n_samples] or shape[n_samples* n_class]
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
        self.silent = silent
        self.objective = objective
        self.nthread = nthread
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
        self._Booster = None
        self.best_iteration = -1
        if callable(self.objective):
            self.fobj = _objective_function_wrapper(self.objective)
        else:
            self.fobj = None

    def booster(self):
        """
        Get the underlying lightgbm Booster of this model.
        This will raise an exception when fit was not called

        Returns
        -------
        booster : a lightgbm booster of underlying model
        """
        if self._Booster is None:
            raise LightGBMError('Need to call fit beforehand')
        return self._Booster

    def get_params(self, deep=False):
        """
        Get parameters
        """
        params = super(LGBMModel, self).get_params(deep=deep)
        if self.nthread <= 0:
            params.pop('nthread', None)
        return params

    def fit(self, X, y,
            sample_weight=None, init_score=None, group=None,
            eval_set=None, eval_sample_weight=None,
            eval_init_score=None, eval_group=None,
            eval_metric=None,
            early_stopping_rounds=None, verbose=True,
            feature_name=None, categorical_feature=None,
            other_params=None):
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
        other_params: dict
            Other parameters

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
        params['verbose'] = 0 if self.silent else 1

        if other_params is not None:
            params.update(other_params)

        if self.fobj:
            params["objective"] = "None"
        else:
            params["objective"] = self.objective
            if eval_metric is None and eval_set is not None:
                eval_metric = {
                    'regression': 'l2',
                    'binary': 'binary_logloss',
                    'lambdarank': 'ndcg',
                    'multiclass': 'multi_logloss'
                }.get(self.objective, None)

        if callable(eval_metric):
            feval = _eval_function_wrapper(eval_metric)
        elif is_str(eval_metric) or isinstance(eval_metric, list):
            feval = None
            params.update({'metric': eval_metric})
        else:
            feval = None

        def _construct_dataset(X, y, sample_weight, init_score, group):
            ret = Dataset(X, label=y, weight=sample_weight, group=group)
            ret.set_init_score(init_score)
            return ret

        train_set = _construct_dataset(X, y, sample_weight, init_score, group)

        valid_sets = []
        if eval_set is not None:
            if isinstance(eval_set, tuple):
                eval_set = [eval_set]
            for i, valid_data in enumerate(eval_set):
                """reduce cost for prediction training data"""
                if valid_data[0] is X and valid_data[1] is y:
                    valid_set = train_set
                else:
                    valid_weight = None if eval_sample_weight is None else eval_sample_weight.get(i, None)
                    valid_init_score = None if eval_init_score is None else eval_init_score.get(i, None)
                    valid_group = None if eval_group is None else eval_group.get(i, None)
                    valid_set = _construct_dataset(valid_data[0], valid_data[1], valid_weight, valid_init_score, valid_group)
                valid_sets.append(valid_set)

        self._Booster = train(params, train_set,
                              self.n_estimators, valid_sets=valid_sets,
                              early_stopping_rounds=early_stopping_rounds,
                              evals_result=evals_result, fobj=self.fobj, feval=feval,
                              verbose_eval=verbose, feature_name=feature_name,
                              categorical_feature=categorical_feature)

        if evals_result:
            for val in evals_result.items():
                evals_result_key = list(val[1].keys())[0]
                evals_result[val[0]][evals_result_key] = val[1][evals_result_key]
            self.evals_result_ = evals_result

        if early_stopping_rounds is not None:
            self.best_iteration = self._Booster.best_iteration
        return self

    def predict(self, data, raw_score=False, num_iteration=0):
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
        return self.booster().predict(data,
                                      raw_score=raw_score,
                                      num_iteration=num_iteration)

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
        return self.booster().predict(X,
                                      pred_leaf=True,
                                      num_iteration=num_iteration)

    def evals_result(self):
        """
        Return the evaluation results.

        Returns
        -------
        evals_result : dictionary
        """
        if self.evals_result_:
            evals_result = self.evals_result_
        else:
            raise LightGBMError('No results found.')

        return evals_result

    def feature_importance(self):
        """
        Feature importances

        Returns
        -------
        Array of normailized feature importances
        """
        importace_array = self._Booster.feature_importance().astype(np.float32)
        return importace_array / importace_array.sum()

class LGBMRegressor(LGBMModel, LGBMRegressorBase):

    def fit(self, X, y,
            sample_weight=None, init_score=None,
            eval_set=None, eval_sample_weight=None,
            eval_init_score=None,
            eval_metric=None,
            early_stopping_rounds=None, verbose=True,
            feature_name=None, categorical_feature=None,
            other_params=None):

        super(LGBMRegressor, self).fit(X, y, sample_weight, init_score, None,
                                       eval_set, eval_sample_weight, eval_init_score, None,
                                       eval_metric, early_stopping_rounds,
                                       verbose, feature_name, categorical_feature,
                                       other_params)
        return self

class LGBMClassifier(LGBMModel, LGBMClassifierBase):

    def __init__(self, boosting_type="gbdt", num_leaves=31, max_depth=-1,
                 learning_rate=0.1, n_estimators=10, max_bin=255,
                 silent=True, objective="binary",
                 nthread=-1, min_split_gain=0, min_child_weight=5, min_child_samples=10,
                 subsample=1, subsample_freq=1, colsample_bytree=1,
                 reg_alpha=0, reg_lambda=0, scale_pos_weight=1,
                 is_unbalance=False, seed=0):
        super(LGBMClassifier, self).__init__(num_leaves, max_depth,
                                             learning_rate, n_estimators, max_bin,
                                             silent, objective, nthread,
                                             min_split_gain, min_child_weight, min_child_samples,
                                             subsample, subsample_freq, colsample_bytree,
                                             reg_alpha, reg_lambda, scale_pos_weight,
                                             is_unbalance, seed)

    def fit(self, X, y,
            sample_weight=None, init_score=None,
            eval_set=None, eval_sample_weight=None,
            eval_init_score=None,
            eval_metric=None,
            early_stopping_rounds=None, verbose=True,
            feature_name=None, categorical_feature=None,
            other_params=None):

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        if other_params is None:
            other_params = {}
        if self.n_classes_ > 2:
            # Switch to using a multiclass objective in the underlying LGBM instance
            self.objective = "multiclass"
            other_params['num_class'] = self.n_classes_

        self._le = LGBMLabelEncoder().fit(y)
        training_labels = self._le.transform(y)

        if eval_set is not None:
            eval_set = list((x[0], self._le.transform(x[1])) for x in eval_set)

        super(LGBMClassifier, self).fit(X, training_labels, sample_weight, init_score, None,
                                        eval_set, eval_sample_weight, eval_init_score, None,
                                        eval_metric, early_stopping_rounds,
                                        verbose, feature_name, categorical_feature,
                                        other_params)
        return self

    def predict(self, data, raw_score=False, num_iteration=0):
        class_probs = self.booster().predict(data,
                                             raw_score=raw_score,
                                             num_iteration=num_iteration)
        if len(class_probs.shape) > 1:
            column_indexes = np.argmax(class_probs, axis=1)
        else:
            column_indexes = np.repeat(0, class_probs.shape[0])
            column_indexes[class_probs > 0.5] = 1
        return self._le.inverse_transform(column_indexes)

    def predict_proba(self, data, raw_score=False, num_iteration=0):
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
        class_probs = self.booster().predict(data,
                                             raw_score=raw_score,
                                             num_iteration=num_iteration)
        if self.n_classes_ > 2:
            return class_probs
        else:
            classone_probs = class_probs
            classzero_probs = 1.0 - classone_probs
            return np.vstack((classzero_probs, classone_probs)).transpose()

class LGBMRanker(LGBMModel):

    def __init__(self, boosting_type="gbdt", num_leaves=31, max_depth=-1,
                 learning_rate=0.1, n_estimators=10, max_bin=255,
                 silent=True, objective="lambdarank",
                 nthread=-1, min_split_gain=0, min_child_weight=5, min_child_samples=10,
                 subsample=1, subsample_freq=1, colsample_bytree=1,
                 reg_alpha=0, reg_lambda=0, scale_pos_weight=1,
                 is_unbalance=False, seed=0):
        super(LGBMRanker, self).__init__(num_leaves, max_depth,
                                         learning_rate, n_estimators, max_bin,
                                         silent, objective, nthread,
                                         min_split_gain, min_child_weight, min_child_samples,
                                         subsample, subsample_freq, colsample_bytree,
                                         reg_alpha, reg_lambda, scale_pos_weight,
                                         is_unbalance, seed)

    def fit(self, X, y,
            sample_weight=None, init_score=None, group=None,
            eval_set=None, eval_sample_weight=None,
            eval_init_score=None, eval_group=None,
            eval_metric=None, eval_at=None,
            early_stopping_rounds=None, verbose=True,
            feature_name=None, categorical_feature=None,
            other_params=None):
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
            else:
                for inner_group in eval_group:
                    if inner_group is None:
                        raise ValueError("Should set group for all eval dataset for ranking task")

        if eval_at is not None:
            other_params = {} if other_params is None else other_params
            other_params['ndcg_eval_at'] = list(eval_at)
        super(LGBMRanker, self).fit(X, y, sample_weight, init_score, group,
                                    eval_set, eval_sample_weight, eval_init_score, eval_group,
                                    eval_metric, early_stopping_rounds,
                                    verbose, feature_name, categorical_feature,
                                    other_params)
        return self
