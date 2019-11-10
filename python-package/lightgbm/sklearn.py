# coding: utf-8
"""Scikit-learn wrapper interface for LightGBM."""
from __future__ import absolute_import

import numpy as np

from .basic import Dataset, LightGBMError, _ConfigAliases
from .compat import (SKLEARN_INSTALLED, _LGBMClassifierBase,
                     LGBMNotFittedError, _LGBMLabelEncoder, _LGBMModelBase,
                     _LGBMRegressorBase, _LGBMCheckXY, _LGBMCheckArray, _LGBMCheckConsistentLength,
                     _LGBMAssertAllFinite, _LGBMCheckClassificationTargets, _LGBMComputeSampleWeight,
                     argc_, range_, zip_, string_type, DataFrame, DataTable)
from .engine import train


class _ObjectiveFunctionWrapper(object):
    """Proxy class for objective function."""

    def __init__(self, func):
        """Construct a proxy class.

        This class transforms objective function to match objective function with signature ``new_func(preds, dataset)``
        as expected by ``lightgbm.engine.train``.

        Parameters
        ----------
        func : callable
            Expects a callable with signature ``func(y_true, y_pred)`` or ``func(y_true, y_pred, group)
            and returns (grad, hess):

                y_true : array-like of shape = [n_samples]
                    The target values.
                y_pred : array-like of shape = [n_samples] or shape = [n_samples * n_classes] (for multi-class task)
                    The predicted values.
                group : array-like
                    Group/query data, used for ranking task.
                grad : array-like of shape = [n_samples] or shape = [n_samples * n_classes] (for multi-class task)
                    The value of the first order derivative (gradient) for each sample point.
                hess : array-like of shape = [n_samples] or shape = [n_samples * n_classes] (for multi-class task)
                    The value of the second order derivative (Hessian) for each sample point.

        .. note::

            For multi-class task, the y_pred is group by class_id first, then group by row_id.
            If you want to get i-th row y_pred in j-th class, the access way is y_pred[j * num_data + i]
            and you should group grad and hess in this way as well.
        """
        self.func = func

    def __call__(self, preds, dataset):
        """Call passed function with appropriate arguments.

        Parameters
        ----------
        preds : array-like of shape = [n_samples] or shape = [n_samples * n_classes] (for multi-class task)
            The predicted values.
        dataset : Dataset
            The training dataset.

        Returns
        -------
        grad : array-like of shape = [n_samples] or shape = [n_samples * n_classes] (for multi-class task)
            The value of the first order derivative (gradient) for each sample point.
        hess : array-like of shape = [n_samples] or shape = [n_samples * n_classes] (for multi-class task)
            The value of the second order derivative (Hessian) for each sample point.
        """
        labels = dataset.get_label()
        argc = argc_(self.func)
        if argc == 2:
            grad, hess = self.func(labels, preds)
        elif argc == 3:
            grad, hess = self.func(labels, preds, dataset.get_group())
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


class _EvalFunctionWrapper(object):
    """Proxy class for evaluation function."""

    def __init__(self, func):
        """Construct a proxy class.

        This class transforms evaluation function to match evaluation function with signature ``new_func(preds, dataset)``
        as expected by ``lightgbm.engine.train``.

        Parameters
        ----------
        func : callable
            Expects a callable with following signatures:
            ``func(y_true, y_pred)``,
            ``func(y_true, y_pred, weight)``
            or ``func(y_true, y_pred, weight, group)``
            and returns (eval_name, eval_result, is_higher_better) or
            list of (eval_name, eval_result, is_higher_better):

                y_true : array-like of shape = [n_samples]
                    The target values.
                y_pred : array-like of shape = [n_samples] or shape = [n_samples * n_classes] (for multi-class task)
                    The predicted values.
                weight : array-like of shape = [n_samples]
                    The weight of samples.
                group : array-like
                    Group/query data, used for ranking task.
                eval_name : string
                    The name of evaluation function (without whitespaces).
                eval_result : float
                    The eval result.
                is_higher_better : bool
                    Is eval result higher better, e.g. AUC is ``is_higher_better``.

        .. note::

            For multi-class task, the y_pred is group by class_id first, then group by row_id.
            If you want to get i-th row y_pred in j-th class, the access way is y_pred[j * num_data + i].
        """
        self.func = func

    def __call__(self, preds, dataset):
        """Call passed function with appropriate arguments.

        Parameters
        ----------
        preds : array-like of shape = [n_samples] or shape = [n_samples * n_classes] (for multi-class task)
            The predicted values.
        dataset : Dataset
            The training dataset.

        Returns
        -------
        eval_name : string
            The name of evaluation function (without whitespaces).
        eval_result : float
            The eval result.
        is_higher_better : bool
            Is eval result higher better, e.g. AUC is ``is_higher_better``.
        """
        labels = dataset.get_label()
        argc = argc_(self.func)
        if argc == 2:
            return self.func(labels, preds)
        elif argc == 3:
            return self.func(labels, preds, dataset.get_weight())
        elif argc == 4:
            return self.func(labels, preds, dataset.get_weight(), dataset.get_group())
        else:
            raise TypeError("Self-defined eval function should have 2, 3 or 4 arguments, got %d" % argc)


class LGBMModel(_LGBMModelBase):
    """Implementation of the scikit-learn API for LightGBM."""

    def __init__(self, boosting_type='gbdt', num_leaves=31, max_depth=-1,
                 learning_rate=0.1, n_estimators=100,
                 subsample_for_bin=200000, objective=None, class_weight=None,
                 min_split_gain=0., min_child_weight=1e-3, min_child_samples=20,
                 subsample=1., subsample_freq=0, colsample_bytree=1.,
                 reg_alpha=0., reg_lambda=0., random_state=None,
                 n_jobs=-1, silent=True, importance_type='split', **kwargs):
        r"""Construct a gradient boosting model.

        Parameters
        ----------
        boosting_type : string, optional (default='gbdt')
            'gbdt', traditional Gradient Boosting Decision Tree.
            'dart', Dropouts meet Multiple Additive Regression Trees.
            'goss', Gradient-based One-Side Sampling.
            'rf', Random Forest.
        num_leaves : int, optional (default=31)
            Maximum tree leaves for base learners.
        max_depth : int, optional (default=-1)
            Maximum tree depth for base learners, <=0 means no limit.
        learning_rate : float, optional (default=0.1)
            Boosting learning rate.
            You can use ``callbacks`` parameter of ``fit`` method to shrink/adapt learning rate
            in training using ``reset_parameter`` callback.
            Note, that this will ignore the ``learning_rate`` argument in training.
        n_estimators : int, optional (default=100)
            Number of boosted trees to fit.
        subsample_for_bin : int, optional (default=200000)
            Number of samples for constructing bins.
        objective : string, callable or None, optional (default=None)
            Specify the learning task and the corresponding learning objective or
            a custom objective function to be used (see note below).
            Default: 'regression' for LGBMRegressor, 'binary' or 'multiclass' for LGBMClassifier, 'lambdarank' for LGBMRanker.
        class_weight : dict, 'balanced' or None, optional (default=None)
            Weights associated with classes in the form ``{class_label: weight}``.
            Use this parameter only for multi-class classification task;
            for binary classification task you may use ``is_unbalance`` or ``scale_pos_weight`` parameters.
            Note, that the usage of all these parameters will result in poor estimates of the individual class probabilities.
            You may want to consider performing probability calibration
            (https://scikit-learn.org/stable/modules/calibration.html) of your model.
            The 'balanced' mode uses the values of y to automatically adjust weights
            inversely proportional to class frequencies in the input data as ``n_samples / (n_classes * np.bincount(y))``.
            If None, all classes are supposed to have weight one.
            Note, that these weights will be multiplied with ``sample_weight`` (passed through the ``fit`` method)
            if ``sample_weight`` is specified.
        min_split_gain : float, optional (default=0.)
            Minimum loss reduction required to make a further partition on a leaf node of the tree.
        min_child_weight : float, optional (default=1e-3)
            Minimum sum of instance weight (hessian) needed in a child (leaf).
        min_child_samples : int, optional (default=20)
            Minimum number of data needed in a child (leaf).
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
            If None, default seeds in C++ code will be used.
        n_jobs : int, optional (default=-1)
            Number of parallel threads.
        silent : bool, optional (default=True)
            Whether to print messages while running boosting.
        importance_type : string, optional (default='split')
            The type of feature importance to be filled into ``feature_importances_``.
            If 'split', result contains numbers of times the feature is used in a model.
            If 'gain', result contains total gains of splits which use the feature.
        **kwargs
            Other parameters for the model.
            Check http://lightgbm.readthedocs.io/en/latest/Parameters.html for more parameters.

            .. warning::

                \*\*kwargs is not supported in sklearn, it may cause unexpected issues.

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
        A custom objective function can be provided for the ``objective`` parameter.
        In this case, it should have the signature
        ``objective(y_true, y_pred) -> grad, hess`` or
        ``objective(y_true, y_pred, group) -> grad, hess``:

            y_true : array-like of shape = [n_samples]
                The target values.
            y_pred : array-like of shape = [n_samples] or shape = [n_samples * n_classes] (for multi-class task)
                The predicted values.
            group : array-like
                Group/query data, used for ranking task.
            grad : array-like of shape = [n_samples] or shape = [n_samples * n_classes] (for multi-class task)
                The value of the first order derivative (gradient) for each sample point.
            hess : array-like of shape = [n_samples] or shape = [n_samples * n_classes] (for multi-class task)
                The value of the second order derivative (Hessian) for each sample point.

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
        self.importance_type = importance_type
        self._Booster = None
        self._evals_result = None
        self._best_score = None
        self._best_iteration = None
        self._other_params = {}
        self._objective = objective
        self.class_weight = class_weight
        self._class_weight = None
        self._class_map = None
        self._n_features = None
        self._classes = None
        self._n_classes = None
        self.set_params(**kwargs)

    def _more_tags(self):
        return {'allow_nan': True,
                'X_types': ['2darray', 'sparse', '1dlabels']}

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, optional (default=True)
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        params = super(LGBMModel, self).get_params(deep=deep)
        params.update(self._other_params)
        return params

    # minor change to support `**kwargs`
    def set_params(self, **params):
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params
            Parameter names with their new values.

        Returns
        -------
        self : object
            Returns self.
        """
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
            A list of (X, y) tuple pairs to use as validation sets.
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
            If callable, it should be a custom evaluation metric, see note below for more details.
            In either case, the ``metric`` from the model parameters will be evaluated and used as well.
            Default: 'l2' for LGBMRegressor, 'logloss' for LGBMClassifier, 'ndcg' for LGBMRanker.
        early_stopping_rounds : int or None, optional (default=None)
            Activates early stopping. The model will train until the validation score stops improving.
            Validation score needs to improve at least every ``early_stopping_rounds`` round(s)
            to continue training.
            Requires at least one validation data and one metric.
            If there's more than one, will check all of them. But the training data is ignored anyway.
            To check only the first metric, set the ``first_metric_only`` parameter to ``True``
            in additional parameters ``**kwargs`` of the model constructor.
        verbose : bool or int, optional (default=True)
            Requires at least one evaluation data.
            If True, the eval metric on the eval set is printed at each boosting stage.
            If int, the eval metric on the eval set is printed at every ``verbose`` boosting stage.
            The last boosting stage or the boosting stage found by using ``early_stopping_rounds`` is also printed.

            .. rubric:: Example

            With ``verbose`` = 4 and at least one item in ``eval_set``,
            an evaluation metric is printed every 4 (instead of 1) boosting stages.

        feature_name : list of strings or 'auto', optional (default='auto')
            Feature names.
            If 'auto' and data is pandas DataFrame, data columns names are used.
        categorical_feature : list of strings or int, or 'auto', optional (default='auto')
            Categorical features.
            If list of int, interpreted as indices.
            If list of strings, interpreted as feature names (need to specify ``feature_name`` as well).
            If 'auto' and data is pandas DataFrame, pandas unordered categorical columns are used.
            All values in categorical features should be less than int32 max value (2147483647).
            Large values could be memory consuming. Consider using consecutive integers starting from zero.
            All negative values in categorical features will be treated as missing values.
            The output cannot be monotonically constrained with respect to a categorical feature.
        callbacks : list of callback functions or None, optional (default=None)
            List of callback functions that are applied at each iteration.
            See Callbacks in Python API for more information.

        Returns
        -------
        self : object
            Returns self.

        Note
        ----
        Custom eval function expects a callable with following signatures:
        ``func(y_true, y_pred)``, ``func(y_true, y_pred, weight)`` or
        ``func(y_true, y_pred, weight, group)``
        and returns (eval_name, eval_result, is_higher_better) or
        list of (eval_name, eval_result, is_higher_better):

            y_true : array-like of shape = [n_samples]
                The target values.
            y_pred : array-like of shape = [n_samples] or shape = [n_samples * n_classes] (for multi-class task)
                The predicted values.
            weight : array-like of shape = [n_samples]
                The weight of samples.
            group : array-like
                Group/query data, used for ranking task.
            eval_name : string
                The name of evaluation function (without whitespaces).
            eval_result : float
                The eval result.
            is_higher_better : bool
                Is eval result higher better, e.g. AUC is ``is_higher_better``.

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
            self._fobj = _ObjectiveFunctionWrapper(self._objective)
        else:
            self._fobj = None
        evals_result = {}
        params = self.get_params()
        # user can set verbose with kwargs, it has higher priority
        if not any(verbose_alias in params for verbose_alias in _ConfigAliases.get("verbosity")) and self.silent:
            params['verbose'] = -1
        params.pop('silent', None)
        params.pop('importance_type', None)
        params.pop('n_estimators', None)
        params.pop('class_weight', None)
        for alias in _ConfigAliases.get('objective'):
            params.pop(alias, None)
        if self._n_classes is not None and self._n_classes > 2:
            for alias in _ConfigAliases.get('num_class'):
                params.pop(alias, None)
            params['num_class'] = self._n_classes
        if hasattr(self, '_eval_at'):
            for alias in _ConfigAliases.get('eval_at'):
                params.pop(alias, None)
            params['eval_at'] = self._eval_at
        params['objective'] = self._objective
        if self._fobj:
            params['objective'] = 'None'  # objective = nullptr for unknown objective

        if callable(eval_metric):
            feval = _EvalFunctionWrapper(eval_metric)
        else:
            feval = None
            # register default metric for consistency with callable eval_metric case
            original_metric = self._objective if isinstance(self._objective, string_type) else None
            if original_metric is None:
                # try to deduce from class instance
                if isinstance(self, LGBMRegressor):
                    original_metric = "l2"
                elif isinstance(self, LGBMClassifier):
                    original_metric = "multi_logloss" if self._n_classes > 2 else "binary_logloss"
                elif isinstance(self, LGBMRanker):
                    original_metric = "ndcg"
            # overwrite default metric by explicitly set metric
            for metric_alias in _ConfigAliases.get("metric"):
                if metric_alias in params:
                    original_metric = params.pop(metric_alias)
            # concatenate metric from params (or default if not provided in params) and eval_metric
            original_metric = [original_metric] if isinstance(original_metric, (string_type, type(None))) else original_metric
            eval_metric = [eval_metric] if isinstance(eval_metric, (string_type, type(None))) else eval_metric
            params['metric'] = [e for e in eval_metric if e not in original_metric] + original_metric
            params['metric'] = [metric for metric in params['metric'] if metric is not None]

        if not isinstance(X, (DataFrame, DataTable)):
            _X, _y = _LGBMCheckXY(X, y, accept_sparse=True, force_all_finite=False, ensure_min_samples=2)
            _LGBMCheckConsistentLength(_X, _y, sample_weight)
        else:
            _X, _y = X, y

        if self._class_weight is None:
            self._class_weight = self.class_weight
        if self._class_weight is not None:
            class_sample_weight = _LGBMComputeSampleWeight(self._class_weight, y)
            if sample_weight is None or len(sample_weight) == 0:
                sample_weight = class_sample_weight
            else:
                sample_weight = np.multiply(sample_weight, class_sample_weight)

        self._n_features = _X.shape[1]

        def _construct_dataset(X, y, sample_weight, init_score, group, params):
            ret = Dataset(X, label=y, weight=sample_weight, group=group, params=params)
            return ret.set_init_score(init_score)

        train_set = _construct_dataset(_X, _y, sample_weight, init_score, group, params)

        valid_sets = []
        if eval_set is not None:

            def _get_meta_data(collection, name, i):
                if collection is None:
                    return None
                elif isinstance(collection, list):
                    return collection[i] if len(collection) > i else None
                elif isinstance(collection, dict):
                    return collection.get(i, None)
                else:
                    raise TypeError('{} should be dict or list'.format(name))

            if isinstance(eval_set, tuple):
                eval_set = [eval_set]
            for i, valid_data in enumerate(eval_set):
                # reduce cost for prediction training data
                if valid_data[0] is X and valid_data[1] is y:
                    valid_set = train_set
                else:
                    valid_weight = _get_meta_data(eval_sample_weight, 'eval_sample_weight', i)
                    valid_class_weight = _get_meta_data(eval_class_weight, 'eval_class_weight', i)
                    if valid_class_weight is not None:
                        if isinstance(valid_class_weight, dict) and self._class_map is not None:
                            valid_class_weight = {self._class_map[k]: v for k, v in valid_class_weight.items()}
                        valid_class_sample_weight = _LGBMComputeSampleWeight(valid_class_weight, valid_data[1])
                        if valid_weight is None or len(valid_weight) == 0:
                            valid_weight = valid_class_sample_weight
                        else:
                            valid_weight = np.multiply(valid_weight, valid_class_sample_weight)
                    valid_init_score = _get_meta_data(eval_init_score, 'eval_init_score', i)
                    valid_group = _get_meta_data(eval_group, 'eval_group', i)
                    valid_set = _construct_dataset(valid_data[0], valid_data[1],
                                                   valid_weight, valid_init_score, valid_group, params)
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

    def predict(self, X, raw_score=False, num_iteration=None,
                pred_leaf=False, pred_contrib=False, **kwargs):
        """Return the predicted value for each sample.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            Input features matrix.
        raw_score : bool, optional (default=False)
            Whether to predict raw scores.
        num_iteration : int or None, optional (default=None)
            Limit number of iterations in the prediction.
            If None, if the best iteration exists, it is used; otherwise, all trees are used.
            If <= 0, all trees are used (no limits).
        pred_leaf : bool, optional (default=False)
            Whether to predict leaf index.
        pred_contrib : bool, optional (default=False)
            Whether to predict feature contributions.

            .. note::

                If you want to get more explanations for your model's predictions using SHAP values,
                like SHAP interaction values,
                you can install the shap package (https://github.com/slundberg/shap).
                Note that unlike the shap package, with ``pred_contrib`` we return a matrix with an extra
                column, where the last column is the expected value.

        **kwargs
            Other parameters for the prediction.

        Returns
        -------
        predicted_result : array-like of shape = [n_samples] or shape = [n_samples, n_classes]
            The predicted values.
        X_leaves : array-like of shape = [n_samples, n_trees] or shape = [n_samples, n_trees * n_classes]
            If ``pred_leaf=True``, the predicted leaf of every tree for each sample.
        X_SHAP_values : array-like of shape = [n_samples, n_features + 1] or shape = [n_samples, (n_features + 1) * n_classes]
            If ``pred_contrib=True``, the feature contributions for each sample.
        """
        if self._n_features is None:
            raise LGBMNotFittedError("Estimator not fitted, call `fit` before exploiting the model.")
        if not isinstance(X, (DataFrame, DataTable)):
            X = _LGBMCheckArray(X, accept_sparse=True, force_all_finite=False)
        n_features = X.shape[1]
        if self._n_features != n_features:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features_ is %s and "
                             "input n_features is %s "
                             % (self._n_features, n_features))
        return self.booster_.predict(X, raw_score=raw_score, num_iteration=num_iteration,
                                     pred_leaf=pred_leaf, pred_contrib=pred_contrib, **kwargs)

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

        .. note::

            Feature importance in sklearn interface used to normalize to 1,
            it's deprecated after 2.0.4 and is the same as Booster.feature_importance() now.
            ``importance_type`` attribute is passed to the function
            to configure the type of importance values to be extracted.
        """
        if self._n_features is None:
            raise LGBMNotFittedError('No feature_importances found. Need to call fit beforehand.')
        return self.booster_.feature_importance(importance_type=self.importance_type)


class LGBMRegressor(LGBMModel, _LGBMRegressorBase):
    """LightGBM regressor."""

    def fit(self, X, y,
            sample_weight=None, init_score=None,
            eval_set=None, eval_names=None, eval_sample_weight=None,
            eval_init_score=None, eval_metric=None, early_stopping_rounds=None,
            verbose=True, feature_name='auto', categorical_feature='auto', callbacks=None):
        """Docstring is inherited from the LGBMModel."""
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


class LGBMClassifier(LGBMModel, _LGBMClassifierBase):
    """LightGBM classifier."""

    def fit(self, X, y,
            sample_weight=None, init_score=None,
            eval_set=None, eval_names=None, eval_sample_weight=None,
            eval_class_weight=None, eval_init_score=None, eval_metric=None,
            early_stopping_rounds=None, verbose=True,
            feature_name='auto', categorical_feature='auto', callbacks=None):
        """Docstring is inherited from the LGBMModel."""
        _LGBMAssertAllFinite(y)
        _LGBMCheckClassificationTargets(y)
        self._le = _LGBMLabelEncoder().fit(y)
        _y = self._le.transform(y)
        self._class_map = dict(zip_(self._le.classes_, self._le.transform(self._le.classes_)))
        if isinstance(self.class_weight, dict):
            self._class_weight = {self._class_map[k]: v for k, v in self.class_weight.items()}

        self._classes = self._le.classes_
        self._n_classes = len(self._classes)
        if self._n_classes > 2:
            # Switch to using a multiclass objective in the underlying LGBM instance
            ova_aliases = ("multiclassova", "multiclass_ova", "ova", "ovr")
            if self._objective not in ova_aliases and not callable(self._objective):
                self._objective = "multiclass"
            if eval_metric in ('logloss', 'binary_logloss'):
                eval_metric = "multi_logloss"
            elif eval_metric in ('error', 'binary_error'):
                eval_metric = "multi_error"
        else:
            if eval_metric in ('logloss', 'multi_logloss'):
                eval_metric = 'binary_logloss'
            elif eval_metric in ('error', 'multi_error'):
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

    fit.__doc__ = LGBMModel.fit.__doc__

    def predict(self, X, raw_score=False, num_iteration=None,
                pred_leaf=False, pred_contrib=False, **kwargs):
        """Docstring is inherited from the LGBMModel."""
        result = self.predict_proba(X, raw_score, num_iteration,
                                    pred_leaf, pred_contrib, **kwargs)
        if raw_score or pred_leaf or pred_contrib:
            return result
        else:
            class_index = np.argmax(result, axis=1)
            return self._le.inverse_transform(class_index)

    predict.__doc__ = LGBMModel.predict.__doc__

    def predict_proba(self, X, raw_score=False, num_iteration=None,
                      pred_leaf=False, pred_contrib=False, **kwargs):
        """Return the predicted probability for each class for each sample.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            Input features matrix.
        raw_score : bool, optional (default=False)
            Whether to predict raw scores.
        num_iteration : int or None, optional (default=None)
            Limit number of iterations in the prediction.
            If None, if the best iteration exists, it is used; otherwise, all trees are used.
            If <= 0, all trees are used (no limits).
        pred_leaf : bool, optional (default=False)
            Whether to predict leaf index.
        pred_contrib : bool, optional (default=False)
            Whether to predict feature contributions.

            .. note::

                If you want to get more explanations for your model's predictions using SHAP values,
                like SHAP interaction values,
                you can install the shap package (https://github.com/slundberg/shap).
                Note that unlike the shap package, with ``pred_contrib`` we return a matrix with an extra
                column, where the last column is the expected value.

        **kwargs
            Other parameters for the prediction.

        Returns
        -------
        predicted_probability : array-like of shape = [n_samples, n_classes]
            The predicted probability for each class for each sample.
        X_leaves : array-like of shape = [n_samples, n_trees * n_classes]
            If ``pred_leaf=True``, the predicted leaf of every tree for each sample.
        X_SHAP_values : array-like of shape = [n_samples, (n_features + 1) * n_classes]
            If ``pred_contrib=True``, the feature contributions for each sample.
        """
        result = super(LGBMClassifier, self).predict(X, raw_score, num_iteration,
                                                     pred_leaf, pred_contrib, **kwargs)
        if self._n_classes > 2 or raw_score or pred_leaf or pred_contrib:
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
            eval_init_score=None, eval_group=None, eval_metric=None,
            eval_at=[1], early_stopping_rounds=None, verbose=True,
            feature_name='auto', categorical_feature='auto', callbacks=None):
        """Docstring is inherited from the LGBMModel."""
        # check group data
        if group is None:
            raise ValueError("Should set group for ranking task")

        if eval_set is not None:
            if eval_group is None:
                raise ValueError("Eval_group cannot be None when eval_set is not None")
            elif len(eval_group) != len(eval_set):
                raise ValueError("Length of eval_group should be equal to eval_set")
            elif (isinstance(eval_group, dict)
                  and any(i not in eval_group or eval_group[i] is None for i in range_(len(eval_group)))
                  or isinstance(eval_group, list)
                  and any(group is None for group in eval_group)):
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
    _before_early_stop, _early_stop, _after_early_stop = _base_doc.partition('early_stopping_rounds :')
    fit.__doc__ = (_before_early_stop
                   + 'eval_at : list of int, optional (default=[1])\n'
                   + ' ' * 12 + 'The evaluation positions of the specified metric.\n'
                   + ' ' * 8 + _early_stop + _after_early_stop)
