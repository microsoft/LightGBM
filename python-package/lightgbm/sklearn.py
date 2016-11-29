"""Scikit-Learn Wrapper interface for LightGBM."""
from __future__ import absolute_import

import numpy as np
from .basic import LightGBMError, Predictor, Dataset, Booster, is_str
from .engine import train
# sklearn
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

def _objective_decorator(func):
    """Decorate an objective function

    Converts an objective function using the typical sklearn metrics to LightGBM ffobj

    Note: for multi-class task, the label/pred is group by class_id first, then group by row_id
          if you want to get i-th row label/pred in j-th class, the access way is label/pred[j*num_data+i]
          and you should group grad and hess in this way as well
    Parameters
    ----------
    func: callable
        Expects a callable with signature ``func(y_true, y_pred)``:

        y_true: array_like of shape [n_samples]
            The target values
        y_pred: array_like of shape [n_samples]
            The predicted values

    Returns
    -------
    new_func: callable
        The new objective function as expected by ``lightgbm.engine.train``.
        The signature is ``new_func(preds, dataset)``:

        preds: array_like, shape [n_samples]
            The predicted values
        dataset: ``dataset``
            The training set from which the labels will be extracted using
            ``dataset.get_label()``
    """
    def inner(preds, dataset):
        """internal function"""
        labels = dataset.get_label()
        return func(labels, preds)
    return inner

class LGBMModel(LGBMModelBase):
    """Implementation of the Scikit-Learn API for LightGBM.

    Parameters
    ----------
    num_leaves : int
        Maximum tree leaves for base learners.
    max_depth : int
        Maximum tree depth for base learners, -1 means not limit. 
    learning_rate : float
        Boosting learning rate 
    n_estimators : int
        Number of boosted trees to fit.
    silent : boolean
        Whether to print messages while running boosting.
    objective : string or callable
        Specify the learning task and the corresponding learning objective or
        a custom objective function to be used (see note below).
    num_class: int
        only affect for multi-class training.
    nthread : int
        Number of parallel threads 
    gamma : float
        Minimum loss reduction required to make a further partition on a leaf node of the tree.
    min_child_weight : int
        Minimum sum of instance weight(hessian) needed in a child.
    subsample : float
        Subsample ratio of the training instance.
    subsample_freq : int
        frequence of subsample, <=0 means no enable
    colsample_bytree : float
        Subsample ratio of columns when constructing each tree.
    colsample_byleaf : float
        Subsample ratio of columns when constructing each leaf.
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
    ``objective(y_true, y_pred) -> grad, hess``:

    y_true: array_like of shape [n_samples]
        The target values
    y_pred: array_like of shape [n_samples]
        The predicted values

    grad: array_like of shape [n_samples]
        The value of the gradient for each sample point.
    hess: array_like of shape [n_samples]
        The value of the second derivative for each sample point

    for multi-class task, the label/pred is group by class_id first, then group by row_id
          if you want to get i-th row label/pred in j-th class, the access way is label/pred[j*num_data+i]
          and you should group grad and hess in this way as well
    """

    def __init__(self, num_leaves=63, max_depth=-1, 
                 learning_rate=0.1, n_estimators=100, max_bin=255,
                 silent=True, objective="regression", num_class=1, 
                 nthread=-1, gamma=0, min_child_weight=1,
                 subsample=1, subsample_freq=1, colsample_bytree=1, colsample_byleaf=1, 
                 reg_alpha=0, reg_lambda=0, scale_pos_weight=1, 
                 is_unbalance=False, seed=0):
        if not SKLEARN_INSTALLED:
            raise LightGBMError('sklearn needs to be installed in order to use this module')

        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_bin = max_bin
        self.silent = silent
        self.objective = objective
        self.num_class = num_class
        self.nthread = nthread
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.subsample_freq = subsample_freq
        self.colsample_bytree = colsample_bytree
        self.colsample_byleaf = colsample_byleaf
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        self.is_unbalance = is_unbalance
        self.seed = seed
        self._Booster = None

    def booster(self):
        """Get the underlying lightgbm Booster of this model.

        This will raise an exception when fit was not called

        Returns
        -------
        booster : a lightgbm booster of underlying model
        """
        if self._Booster is None:
            raise LightGBMError('need to call fit beforehand')
        return self._Booster

    def get_params(self, deep=False):
        """Get parameter.s"""
        params = super(LGBMModel, self).get_params(deep=deep)
        params['verbose'] = 0 if self.silent else 1
        if self.nthread <= 0:
            params.pop('nthread', None)
        return params

    def fit(self, X, y, eval_set=None, eval_metric=None,
            early_stopping_rounds=None, verbose=True):
        """
        Fit the gradient boosting model

        Parameters
        ----------
        X : array_like
            Feature matrix
        y : array_like
            Labels
        eval_set : list, optional
            A list of (X, y) tuple pairs to use as a validation set for early-stopping
        eval_metric : str, list of str, callable, optional
            If a str, should be a built-in evaluation metric to use. See
            doc/parameter.md. If callable, a custom evaluation metric. The call
            signature is func(y_predicted, y_true) where y_true will be a
            Dataset fobject such that you may need to call the get_label
            method. And it must return (eval_name, feature_result, is_bigger_better)
        early_stopping_rounds : int
        verbose : bool
            If `verbose` and an evaluation set is used, writes the evaluation
            metric measured on the validation set to stderr.
        """
        evals_result = {}
        params = self.get_params()

        if callable(self.objective):
            fobj = _objective_decorator(self.objective)
            params["objective"] = "None"
        else:
            fobj = None
        if callable(eval_metric):
            feval = eval_metric
        else:
            feval = None
            params.update({'metric': eval_metric})
        feval = eval_metric if callable(eval_metric) else None


        self._Booster = train(params, (X, y),
                              self.n_estimators, valid_datas=eval_set,
                              early_stopping_rounds=early_stopping_rounds,
                              evals_result=evals_result, fobj=fobj, feval=feval,
                              verbose_eval=verbose)

        if evals_result:
            for val in evals_result.items():
                evals_result_key = list(val[1].keys())[0]
                evals_result[val[0]][evals_result_key] = val[1][evals_result_key]
            self.evals_result_ = evals_result

        if early_stopping_rounds is not None:
            self.best_iteration = self._Booster.best_iteration
        return self

    def predict(self, data, raw_score=False, num_iteration=0):
        return self.booster().predict(data,
                                      raw_score=raw_score,
                                      num_iteration=num_iteration)

    def apply(self, X, num_iteration=0):
        """Return the predicted leaf every tree for each sample.

        Parameters
        ----------
        X : array_like, shape=[n_samples, n_features]
            Input features matrix.

        ntree_limit : int
            Limit number of trees in the prediction; defaults to 0 (use all trees).

        Returns
        -------
        X_leaves : array_like, shape=[n_samples, n_trees]
        """
        return self.booster().predict(X,
                                      pred_leaf=True,
                                      num_iteration=num_iteration)

    def evals_result(self):
        """Return the evaluation results.
        Returns
        -------
        evals_result : dictionary
        """
        if self.evals_result_:
            evals_result = self.evals_result_
        else:
            raise LightGBMError('No results.')

        return evals_result