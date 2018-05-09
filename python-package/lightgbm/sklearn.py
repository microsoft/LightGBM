# coding: utf-8
# pylint: disable = invalid-name, W0105, C0111, C0301
"""Scikit-Learn Wrapper interface for LightGBM."""
from __future__ import absolute_import

import numpy as np
import warnings

from .basic import Dataset, LightGBMError
from .compat import (SKLEARN_INSTALLED, _LGBMClassifierBase,
                     LGBMNotFittedError, _LGBMLabelEncoder, _LGBMModelBase,
                     _LGBMRegressorBase, _LGBMCheckXY, _LGBMCheckArray, _LGBMCheckConsistentLength,
                     _LGBMCheckClassificationTargets, _LGBMComputeSampleWeight,
                     argc_, range_, DataFrame, LGBMDeprecationWarning)
from .engine import train


def _objective_function_wrapper(func):
    """Decorate an objective function
    Note: for multi-class task, the y_pred is group by class_id first, then group by row_id.
          If you want to get i-th row y_pred in j-th class, the access way is y_pred[j * num_data + i]
          and you should group grad and hess in this way as well.

    Parameters
    ----------
    func: callable
        Expects a callable with signature ``func(y_true, y_pred)`` or ``func(y_true, y_pred, group):
            y_true: array-like of shape = [n_samples]
                The target values.
            y_pred: array-like of shape = [n_samples] or shape = [n_samples * n_classes] (for multi-class)
                The predicted values.
            group: array-like
                Group/query data, used for ranking task.

    Returns
    -------
    new_func: callable
        The new objective function as expected by ``lightgbm.engine.train``.
        The signature is ``new_func(preds, dataset)``:

        preds: array-like of shape = [n_samples] or shape = [n_samples * n_classes]
            The predicted values.
        dataset: ``dataset``
            The training set from which the labels will be extracted using
            ``dataset.get_label()``.
    """
    def inner(preds, dataset):
        """internal function"""
        labels = dataset.get_label()
        argc = argc_(func)
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
                for k in range_(num_class):
                    for i in range_(num_data):
                        idx = k * num_data + i
                        grad[idx] *= weight[i]
                        hess[idx] *= weight[i]
        return grad, hess
    return inner


def _eval_function_wrapper(func):
    """Decorate an eval function
    Note: for multi-class task, the y_pred is group by class_id first, then group by row_id.
          If you want to get i-th row y_pred in j-th class, the access way is y_pred[j * num_data + i].

    Parameters
    ----------
    func: callable
        Expects a callable with following functions:
            ``func(y_true, y_pred)``,
            ``func(y_true, y_pred, weight)``
         or ``func(y_true, y_pred, weight, group)``
            and return (eval_name->str, eval_result->float, is_bigger_better->Bool):

            y_true: array-like of shape = [n_samples]
                The target values.
            y_pred: array-like of shape = [n_samples] or shape = [n_samples * n_classes] (for multi-class)
                The predicted values.
            weight: array_like of shape = [n_samples]
                The weight of samples.
            group: array-like
                Group/query data, used for ranking task.

    Returns
    -------
    new_func: callable
        The new eval function as expected by ``lightgbm.engine.train``.
        The signature is ``new_func(preds, dataset)``:

        preds: array-like of shape = [n_samples] or shape = [n_samples * n_classes]
            The predicted values.
        dataset: ``dataset``
            The training set from which the labels will be extracted using
            ``dataset.get_label()``.
    """
    def inner(preds, dataset):
        """internal function"""
        labels = dataset.get_label()
        argc = argc_(func)
        if argc == 2:
            return func(labels, preds)
        elif argc == 3:
            return func(labels, preds, dataset.get_weight())
        elif argc == 4:
            return func(labels, preds, dataset.get_weight(), dataset.get_group())
        else:
            raise TypeError("Self-defined eval function should have 2, 3 or 4 arguments, got %d" % argc)
    return inner


class LGBMModel(_LGBMModelBase):
    """Implementation of the scikit-learn API for LightGBM."""

    def __init__(self, boosting_type="gbdt", num_leaves=31, max_depth=-1,
                 learning_rate=0.1, n_estimators=100,
                 subsample_for_bin=200000, objective=None, class_weight=None,
                 min_split_gain=0., min_child_weight=1e-3, min_child_samples=20,
                 subsample=1., subsample_freq=0, colsample_bytree=1.,
                 reg_alpha=0., reg_lambda=0., random_state=None,
                 n_jobs=-1, silent=True, **kwargs):
        """Construct a gradient boosting model.

        Parameters
        ----------
        boosting_type : string, optional (default="gbdt")
            'gbdt', traditional Gradient Boosting Decision Tree.
            'dart', Dropouts meet Multiple Additive Regression Trees.
            'goss', Gradient-based One-Side Sampling.
            'rf', Random Forest.
        num_leaves : int, optional (default=31)
            Maximum tree leaves for base learners.
        max_depth : int, optional (default=-1)
            Maximum tree depth for base learners, -1 means no limit.
        learning_rate : float, optional (default=0.1)
            Boosting learning rate.
            You can use ``callbacks`` parameter of ``fit`` method to shrink/adapt learning rate
            in training using ``reset_parameter`` callback.
            Note, that this will ignore the ``learning_rate`` argument in training.
        n_estimators : int, optional (default=100)
            Number of boosted trees to fit.
        subsample_for_bin : int, optional (default=50000)
            Number of samples for constructing bins.
        objective : string, callable or None, optional (default=None)
            Specify the learning task and the corresponding learning objective or
            a custom objective function to be used (see note below).
            default: 'regression' for LGBMRegressor, 'binary' or 'multiclass' for LGBMClassifier, 'lambdarank' for LGBMRanker.
        class_weight : dict, 'balanced' or None, optional (default=None)
            Weights associated with classes in the form ``{class_label: weight}``.
            Use this parameter only for multi-class classification task;
            for binary classification task you may use ``is_unbalance`` or ``scale_pos_weight`` parameters.
            The 'balanced' mode uses the values of y to automatically adjust weights
            inversely proportional to class frequencies in the input data as ``n_samples / (n_classes * np.bincount(y))``.
            If None, all classes are supposed to have weight one.
            Note that these weights will be multiplied with ``sample_weight`` (passed through the fit method)
            if ``sample_weight`` is specified.
        min_split_gain : float, optional (default=0.)
            Minimum loss reduction required to make a further partition on a leaf node of the tree.
        min_child_weight : float, optional (default=1e-3)
            Minimum sum of instance weight(hessian) needed in a child(leaf).
        min_child_samples : int, optional (default=20)
            Minimum number of data need in a child(leaf).
        subsample : float, optional (default=1.)
            Subsample ratio of the training instance.
        subsample_freq : int, optional (default=0)
            Frequence of subsample, <=0 means no enable.
        colsample_bytree : float, optional (default=1.)
            Subsample ratio of columns when constructing each tree.
        reg_alpha : float, optional (default=0.)
            L1 regularization term on weights.
        reg_lambda : float, optional (default=0.)
            L2 regularization term on weights.
        random_state : int or None, optional (default=None)
            Random number seed.
            Will use default seeds in c++ code if set to None.
        n_jobs : int, optional (default=-1)
            Number of parallel threads.
        silent : bool, optional (default=True)
            Whether to print messages while running boosting.
        **kwargs : other parameters
            Check http://lightgbm.readthedocs.io/en/latest/Parameters.html for more parameters.

            Note
            ----
            \\*\\*kwargs is not supported in sklearn, it may cause unexpected issues.

        Attributes
        ----------
        n_features_ : int
            The number of features of fitted model.
        classes_ : array of shape = [n_classes]
            The class label array (only for classification problem).
        n_classes_ : int
            The number of classes (only for classification problem).
        best_score_ : dict or None
            The best score of fitted model.
        best_iteration_ : int or None
            The best iteration of fitted model if ``early_stopping_rounds`` has been specified.
        objective_ : string or callable
            The concrete objective used while fitting this model.
        booster_ : Booster
            The underlying Booster of this model.
        evals_result_ : dict or None
            The evaluation results if ``early_stopping_rounds`` has been specified.
        feature_importances_ : array of shape = [n_features]
            The feature importances (the higher, the more important the feature).

        Note
        ----
        A custom objective function can be provided for the ``objective``
        parameter. In this case, it should have the signature
        ``objective(y_true, y_pred) -> grad, hess`` or
        ``objective(y_true, y_pred, group) -> grad, hess``:

            y_true: array-like of shape = [n_samples]
                The target values.
            y_pred: array-like of shape = [n_samples] or shape = [n_samples * n_classes] (for multi-class task)
                The predicted values.
            group: array-like
                Group/query data, used for ranking task.
            grad: array-like of shape = [n_samples] or shape = [n_samples * n_classes] (for multi-class task)
                The value of the gradient for each sample point.
            hess: array-like of shape = [n_samples] or shape = [n_samples * n_classes] (for multi-class task)
                The value of the second derivative for each sample point.

        For multi-class task, the y_pred is group by class_id first, then group by row_id.
        If you want to get i-th row y_pred in j-th class, the access way is y_pred[j * num_data + i]
        and you should group grad and hess in this way as well.
        """
        if not SKLEARN_INSTALLED:
            raise LightGBMError('Scikit-learn is required for this module')

        self.boosting_type = boosting_type
        self.objective = objective
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample_for_bin = subsample_for_bin
        self.min_split_gain = min_split_gain
        self.min_child_weight = min_child_weight
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.subsample_freq = subsample_freq
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.silent = silent
        self._Booster = None
        self._evals_result = None
        self._best_score = None
        self._best_iteration = None
        self._other_params = {}
        self._objective = objective
        self.class_weight = class_weight
        self._n_features = None
        self._classes = None
        self._n_classes = None
        self.set_params(**kwargs)

    def get_params(self, deep=True):
        params = super(LGBMModel, self).get_params(deep=deep)
        params.update(self._other_params)
        return params

    # minor change to support `**kwargs`
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
            if hasattr(self, '_' + key):
                setattr(self, '_' + key, value)
            self._other_params[key] = value
        return self

    def fit(self, X, y,
            sample_weight=None, init_score=None, group=None,
            eval_set=None, eval_names=None, eval_sample_weight=None,
            eval_class_weight=None, eval_init_score=None, eval_group=None,
            eval_metric=None, early_stopping_rounds=None, verbose=True,
            feature_name='auto', categorical_feature='auto', callbacks=None):
        """Build a gradient boosting model from the training set (X, y).

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            Input feature matrix.
        y : array-like of shape = [n_samples]
            The target values (class labels in classification, real numbers in regression).
        sample_weight : array-like of shape = [n_samples] or None, optional (default=None)
            Weights of training data.
        init_score : array-like of shape = [n_samples] or None, optional (default=None)
            Init score of training data.
        group : array-like or None, optional (default=None)
            Group data of training data.
        eval_set : list or None, optional (default=None)
            A list of (X, y) tuple pairs to use as a validation sets for early-stopping.
        eval_names : list of strings or None, optional (default=None)
            Names of eval_set.
        eval_sample_weight : list of arrays or None, optional (default=None)
            Weights of eval data.
        eval_class_weight : list or None, optional (default=None)
            Class weights of eval data.
        eval_init_score : list of arrays or None, optional (default=None)
            Init score of eval data.
        eval_group : list of arrays or None, optional (default=None)
            Group data of eval data.
        eval_metric : string, list of strings, callable or None, optional (default=None)
            If string, it should be a built-in evaluation metric to use.
            If callable, it should be a custom evaluation metric, see note for more details.
            In either case, the ``metric`` from the model parameters will be evaluated and used as well.
        early_stopping_rounds : int or None, optional (default=None)
            Activates early stopping. The model will train until the validation score stops improving.
            If there's more than one, will check all of them.
            Validation error needs to decrease at least every ``early_stopping_rounds`` round(s)
            to continue training.
        verbose : bool, optional (default=True)
            If True and an evaluation set is used, writes the evaluation progress.
        feature_name : list of strings or 'auto', optional (default="auto")
            Feature names.
            If 'auto' and data is pandas DataFrame, data columns names are used.
        categorical_feature : list of strings or int, or 'auto', optional (default="auto")
            Categorical features.
            If list of int, interpreted as indices.
            If list of strings, interpreted as feature names (need to specify ``feature_name`` as well).
            If 'auto' and data is pandas DataFrame, pandas categorical columns are used.
        callbacks : list of callback functions or None, optional (default=None)
            List of callback functions that are applied at each iteration.
            See Callbacks in Python API for more information.

        Returns
        -------
        self : object
            Returns self.

        Note
        ----
        Custom eval function expects a callable with following functions:
        ``func(y_true, y_pred)``, ``func(y_true, y_pred, weight)`` or
        ``func(y_true, y_pred, weight, group)``.
        Returns (eval_name, eval_result, is_bigger_better) or
        list of (eval_name, eval_result, is_bigger_better)

            y_true: array-like of shape = [n_samples]
                The target values.
            y_pred: array-like of shape = [n_samples] or shape = [n_samples * n_classes] (for multi-class)
                The predicted values.
            weight: array-like of shape = [n_samples]
                The weight of samples.
            group: array-like
                Group/query data, used for ranking task.
            eval_name: str
                The name of evaluation.
            eval_result: float
                The eval result.
            is_bigger_better: bool
                Is eval result bigger better, e.g. AUC is bigger_better.

        For multi-class task, the y_pred is group by class_id first, then group by row_id.
        If you want to get i-th row y_pred in j-th class, the access way is y_pred[j * num_data + i].
        """
        if self._objective is None:
            if isinstance(self, LGBMRegressor):
                self._objective = "regression"
            elif isinstance(self, LGBMClassifier):
                self._objective = "binary"
            elif isinstance(self, LGBMRanker):
                self._objective = "lambdarank"
            else:
                raise ValueError("Unknown LGBMModel type.")
        if callable(self._objective):
            self._fobj = _objective_function_wrapper(self._objective)
        else:
            self._fobj = None
        evals_result = {}
        params = self.get_params()
        # sklearn interface has another naming convention
        params.setdefault('seed', params.pop('random_state'))
        params.setdefault('nthread', params.pop('n_jobs'))
        # user can set verbose with kwargs, it has higher priority
        if 'verbose' not in params and self.silent:
            params['verbose'] = 0
        params.pop('silent', None)
        params.pop('n_estimators', None)
        params.pop('class_weight', None)
        if self._n_classes is not None and self._n_classes > 2:
            params['num_class'] = self._n_classes
        if hasattr(self, '_eval_at'):
            params['ndcg_eval_at'] = self._eval_at
        params['objective'] = self._objective
        if self._fobj:
            params['objective'] = 'None'  # objective = nullptr for unknown objective

        if callable(eval_metric):
            feval = _eval_function_wrapper(eval_metric)
        else:
            feval = None
            params['metric'] = eval_metric

        if not isinstance(X, DataFrame):
            X, y = _LGBMCheckXY(X, y, accept_sparse=True, force_all_finite=False, ensure_min_samples=2)
            _LGBMCheckConsistentLength(X, y, sample_weight)

        if self.class_weight is not None:
            class_sample_weight = _LGBMComputeSampleWeight(self.class_weight, y)
            if sample_weight is None or len(sample_weight) == 0:
                sample_weight = class_sample_weight
            else:
                sample_weight = np.multiply(sample_weight, class_sample_weight)

        self._n_features = X.shape[1]

        def _construct_dataset(X, y, sample_weight, init_score, group, params):
            ret = Dataset(X, label=y, weight=sample_weight, group=group, params=params)
            ret.set_init_score(init_score)
            return ret

        train_set = _construct_dataset(X, y, sample_weight, init_score, group, params)

        valid_sets = []
        if eval_set is not None:
            if isinstance(eval_set, tuple):
                eval_set = [eval_set]
            for i, valid_data in enumerate(eval_set):
                # reduce cost for prediction training data
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
                            raise TypeError('eval_sample_weight, eval_class_weight, eval_init_score, and eval_group should be dict or list')
                    valid_weight = get_meta_data(eval_sample_weight, i)
                    if get_meta_data(eval_class_weight, i) is not None:
                        valid_class_sample_weight = _LGBMComputeSampleWeight(get_meta_data(eval_class_weight, i), valid_data[1])
                        if valid_weight is None or len(valid_weight) == 0:
                            valid_weight = valid_class_sample_weight
                        else:
                            valid_weight = np.multiply(valid_weight, valid_class_sample_weight)
                    valid_init_score = get_meta_data(eval_init_score, i)
                    valid_group = get_meta_data(eval_group, i)
                    valid_set = _construct_dataset(valid_data[0], valid_data[1], valid_weight, valid_init_score, valid_group, params)
                valid_sets.append(valid_set)

        self._Booster = train(params, train_set,
                              self.n_estimators, valid_sets=valid_sets, valid_names=eval_names,
                              early_stopping_rounds=early_stopping_rounds,
                              evals_result=evals_result, fobj=self._fobj, feval=feval,
                              verbose_eval=verbose, feature_name=feature_name,
                              categorical_feature=categorical_feature,
                              callbacks=callbacks)

        if evals_result:
            self._evals_result = evals_result

        if early_stopping_rounds is not None:
            self._best_iteration = self._Booster.best_iteration

        self._best_score = self._Booster.best_score

        # free dataset
        self.booster_.free_dataset()
        del train_set, valid_sets
        return self

    def predict(self, X, raw_score=False, num_iteration=-1,
                pred_leaf=False, pred_contrib=False, **kwargs):
        """Return the predicted value for each sample.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            Input features matrix.
        raw_score : bool, optional (default=False)
            Whether to predict raw scores.
        num_iteration : int, optional (default=-1)
            Limit number of iterations in the prediction.
            If <= 0, uses all trees (no limits).
        pred_leaf : bool, optional (default=False)
            Whether to predict leaf index.
        pred_contrib : bool, optional (default=False)
            Whether to predict feature contributions.
        **kwargs : other parameters for the prediction

        Returns
        -------
        predicted_result : array-like of shape = [n_samples] or shape = [n_samples, n_classes]
            The predicted values.
        X_leaves : array-like of shape = [n_samples, n_trees] or shape [n_samples, n_trees * n_classes]
            If ``pred_leaf=True``, the predicted leaf every tree for each sample.
        X_SHAP_values : array-like of shape = [n_samples, n_features + 1] or shape [n_samples, (n_features + 1) * n_classes]
            If ``pred_contrib=True``, the each feature contributions for each sample.
        """
        if self._n_features is None:
            raise LGBMNotFittedError("Estimator not fitted, call `fit` before exploiting the model.")
        if not isinstance(X, DataFrame):
            X = _LGBMCheckArray(X, accept_sparse=True, force_all_finite=False)
        n_features = X.shape[1]
        if self._n_features != n_features:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features_ is %s and "
                             "input n_features is %s "
                             % (self._n_features, n_features))
        return self.booster_.predict(X, raw_score=raw_score, num_iteration=num_iteration,
                                     pred_leaf=pred_leaf, pred_contrib=pred_contrib, **kwargs)

    def apply(self, X, num_iteration=0):
        """Return the predicted leaf every tree for each sample.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            Input features matrix.
        num_iteration : int, optional (default=0)
            Limit number of iterations in the prediction; defaults to 0 (use all trees).

        Returns
        -------
        X_leaves : array-like of shape = [n_samples, n_trees]
            The predicted leaf every tree for each sample.
        """
        warnings.warn('apply method is deprecated and will be removed in 2.2 version.\n'
                      'Please use pred_leaf parameter of predict method instead.',
                      LGBMDeprecationWarning)
        if self._n_features is None:
            raise LGBMNotFittedError("Estimator not fitted, call `fit` before exploiting the model.")
        if not isinstance(X, DataFrame):
            X = _LGBMCheckArray(X, accept_sparse=True, force_all_finite=False)
        n_features = X.shape[1]
        if self._n_features != n_features:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features_ is %s and "
                             "input n_features is %s "
                             % (self._n_features, n_features))
        return self.booster_.predict(X, pred_leaf=True, num_iteration=num_iteration)

    @property
    def n_features_(self):
        """Get the number of features of fitted model."""
        if self._n_features is None:
            raise LGBMNotFittedError('No n_features found. Need to call fit beforehand.')
        return self._n_features

    @property
    def best_score_(self):
        """Get the best score of fitted model."""
        if self._n_features is None:
            raise LGBMNotFittedError('No best_score found. Need to call fit beforehand.')
        return self._best_score

    @property
    def best_iteration_(self):
        """Get the best iteration of fitted model."""
        if self._n_features is None:
            raise LGBMNotFittedError('No best_iteration found. Need to call fit with early_stopping_rounds beforehand.')
        return self._best_iteration

    @property
    def objective_(self):
        """Get the concrete objective used while fitting this model."""
        if self._n_features is None:
            raise LGBMNotFittedError('No objective found. Need to call fit beforehand.')
        return self._objective

    @property
    def booster_(self):
        """Get the underlying lightgbm Booster of this model."""
        if self._Booster is None:
            raise LGBMNotFittedError('No booster found. Need to call fit beforehand.')
        return self._Booster

    @property
    def evals_result_(self):
        """Get the evaluation results."""
        if self._n_features is None:
            raise LGBMNotFittedError('No results found. Need to call fit with eval_set beforehand.')
        return self._evals_result

    @property
    def feature_importances_(self):
        """Get feature importances.

        Note
        ----
        Feature importance in sklearn interface used to normalize to 1,
        it's deprecated after 2.0.4 and same as Booster.feature_importance() now.
        """
        if self._n_features is None:
            raise LGBMNotFittedError('No feature_importances found. Need to call fit beforehand.')
        return self.booster_.feature_importance()


class LGBMRegressor(LGBMModel, _LGBMRegressorBase):
    """LightGBM regressor."""

    def fit(self, X, y,
            sample_weight=None, init_score=None,
            eval_set=None, eval_names=None, eval_sample_weight=None,
            eval_init_score=None, eval_metric="l2", early_stopping_rounds=None,
            verbose=True, feature_name='auto', categorical_feature='auto', callbacks=None):

        super(LGBMRegressor, self).fit(X, y, sample_weight=sample_weight,
                                       init_score=init_score, eval_set=eval_set,
                                       eval_names=eval_names,
                                       eval_sample_weight=eval_sample_weight,
                                       eval_init_score=eval_init_score,
                                       eval_metric=eval_metric,
                                       early_stopping_rounds=early_stopping_rounds,
                                       verbose=verbose, feature_name=feature_name,
                                       categorical_feature=categorical_feature,
                                       callbacks=callbacks)
        return self

    _base_doc = LGBMModel.fit.__doc__
    fit.__doc__ = (_base_doc[:_base_doc.find('eval_class_weight :')]
                   + _base_doc[_base_doc.find('eval_init_score :'):])
    _base_doc = fit.__doc__
    fit.__doc__ = (_base_doc[:_base_doc.find('eval_metric :')]
                   + 'eval_metric : string, list of strings, callable or None, optional (default="l2")\n'
                   + _base_doc[_base_doc.find('            If string, it should be a built-in evaluation metric to use.'):])


class LGBMClassifier(LGBMModel, _LGBMClassifierBase):
    """LightGBM classifier."""

    def fit(self, X, y,
            sample_weight=None, init_score=None,
            eval_set=None, eval_names=None, eval_sample_weight=None,
            eval_class_weight=None, eval_init_score=None, eval_metric="logloss",
            early_stopping_rounds=None, verbose=True,
            feature_name='auto', categorical_feature='auto', callbacks=None):
        _LGBMCheckClassificationTargets(y)
        self._le = _LGBMLabelEncoder().fit(y)
        _y = self._le.transform(y)

        self._classes = self._le.classes_
        self._n_classes = len(self._classes)
        if self._n_classes > 2:
            # Switch to using a multiclass objective in the underlying LGBM instance
            ova_aliases = ("multiclassova", "multiclass_ova", "ova", "ovr")
            if self._objective not in ova_aliases and not callable(self._objective):
                self._objective = "multiclass"
            if eval_metric == 'logloss' or eval_metric == 'binary_logloss':
                eval_metric = "multi_logloss"
            elif eval_metric == 'error' or eval_metric == 'binary_error':
                eval_metric = "multi_error"
        else:
            if eval_metric == 'logloss' or eval_metric == 'multi_logloss':
                eval_metric = 'binary_logloss'
            elif eval_metric == 'error' or eval_metric == 'multi_error':
                eval_metric = 'binary_error'

        if eval_set is not None:
            if isinstance(eval_set, tuple):
                eval_set = [eval_set]
            for i, (valid_x, valid_y) in enumerate(eval_set):
                if valid_x is X and valid_y is y:
                    eval_set[i] = (valid_x, _y)
                else:
                    eval_set[i] = (valid_x, self._le.transform(valid_y))

        super(LGBMClassifier, self).fit(X, _y, sample_weight=sample_weight,
                                        init_score=init_score, eval_set=eval_set,
                                        eval_names=eval_names,
                                        eval_sample_weight=eval_sample_weight,
                                        eval_class_weight=eval_class_weight,
                                        eval_init_score=eval_init_score,
                                        eval_metric=eval_metric,
                                        early_stopping_rounds=early_stopping_rounds,
                                        verbose=verbose, feature_name=feature_name,
                                        categorical_feature=categorical_feature,
                                        callbacks=callbacks)
        return self

    _base_doc = LGBMModel.fit.__doc__
    fit.__doc__ = (_base_doc[:_base_doc.find('eval_metric :')]
                   + 'eval_metric : string, list of strings, callable or None, optional (default="logloss")\n'
                   + _base_doc[_base_doc.find('            If string, it should be a built-in evaluation metric to use.'):])

    def predict(self, X, raw_score=False, num_iteration=-1,
                pred_leaf=False, pred_contrib=False, **kwargs):
        result = self.predict_proba(X, raw_score, num_iteration,
                                    pred_leaf, pred_contrib, **kwargs)
        if raw_score or pred_leaf or pred_contrib:
            return result
        else:
            class_index = np.argmax(result, axis=1)
            return self._le.inverse_transform(class_index)

    def predict_proba(self, X, raw_score=False, num_iteration=-1,
                      pred_leaf=False, pred_contrib=False, **kwargs):
        """Return the predicted probability for each class for each sample.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            Input features matrix.
        raw_score : bool, optional (default=False)
            Whether to predict raw scores.
        num_iteration : int, optional (default=-1)
            Limit number of iterations in the prediction.
            If <= 0, uses all trees (no limits).
        pred_leaf : bool, optional (default=False)
            Whether to predict leaf index.
        pred_contrib : bool, optional (default=False)
            Whether to predict feature contributions.
        **kwargs : other parameters for the prediction

        Returns
        -------
        predicted_probability : array-like of shape = [n_samples, n_classes]
            The predicted probability for each class for each sample.
        X_leaves : array-like of shape = [n_samples, n_trees * n_classes]
            If ``pred_leaf=True``, the predicted leaf every tree for each sample.
        X_SHAP_values : array-like of shape = [n_samples, (n_features + 1) * n_classes]
            If ``pred_contrib=True``, the each feature contributions for each sample.
        """
        result = super(LGBMClassifier, self).predict(X, raw_score, num_iteration,
                                                     pred_leaf, pred_contrib, **kwargs)
        if self._n_classes > 2 or pred_leaf or pred_contrib:
            return result
        else:
            return np.vstack((1. - result, result)).transpose()

    @property
    def classes_(self):
        """Get the class label array."""
        if self._classes is None:
            raise LGBMNotFittedError('No classes found. Need to call fit beforehand.')
        return self._classes

    @property
    def n_classes_(self):
        """Get the number of classes."""
        if self._n_classes is None:
            raise LGBMNotFittedError('No classes found. Need to call fit beforehand.')
        return self._n_classes


class LGBMRanker(LGBMModel):
    """LightGBM ranker."""

    def fit(self, X, y,
            sample_weight=None, init_score=None, group=None,
            eval_set=None, eval_names=None, eval_sample_weight=None,
            eval_init_score=None, eval_group=None, eval_metric='ndcg',
            eval_at=[1], early_stopping_rounds=None, verbose=True,
            feature_name='auto', categorical_feature='auto', callbacks=None):
        # check group data
        if group is None:
            raise ValueError("Should set group for ranking task")

        if eval_set is not None:
            if eval_group is None:
                raise ValueError("Eval_group cannot be None when eval_set is not None")
            elif len(eval_group) != len(eval_set):
                raise ValueError("Length of eval_group should be equal to eval_set")
            elif (isinstance(eval_group, dict) and any(i not in eval_group or eval_group[i] is None for i in range_(len(eval_group)))) \
                    or (isinstance(eval_group, list) and any(group is None for group in eval_group)):
                raise ValueError("Should set group for all eval datasets for ranking task; "
                                 "if you use dict, the index should start from 0")

        self._eval_at = eval_at
        super(LGBMRanker, self).fit(X, y, sample_weight=sample_weight,
                                    init_score=init_score, group=group,
                                    eval_set=eval_set, eval_names=eval_names,
                                    eval_sample_weight=eval_sample_weight,
                                    eval_init_score=eval_init_score, eval_group=eval_group,
                                    eval_metric=eval_metric,
                                    early_stopping_rounds=early_stopping_rounds,
                                    verbose=verbose, feature_name=feature_name,
                                    categorical_feature=categorical_feature,
                                    callbacks=callbacks)
        return self

    _base_doc = LGBMModel.fit.__doc__
    fit.__doc__ = (_base_doc[:_base_doc.find('eval_class_weight :')]
                   + _base_doc[_base_doc.find('eval_init_score :'):])
    _base_doc = fit.__doc__
    fit.__doc__ = (_base_doc[:_base_doc.find('eval_metric :')]
                   + 'eval_metric : string, list of strings, callable or None, optional (default="ndcg")\n'
                   + _base_doc[_base_doc.find('            If string, it should be a built-in evaluation metric to use.'):_base_doc.find('early_stopping_rounds :')]
                   + 'eval_at : list of int, optional (default=[1])\n'
                     '            The evaluation positions of NDCG.\n'
                   + _base_doc[_base_doc.find('        early_stopping_rounds :'):])
