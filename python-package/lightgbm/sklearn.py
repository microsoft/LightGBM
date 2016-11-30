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

def _point_wise_objective(func):
    """Decorate an objective function

    Converts an objective function using the typical sklearn metrics to LightGBM fobj

    Note: for multi-class task, the y_pred is group by class_id first, then group by row_id
          if you want to get i-th row y_pred in j-th class, the access way is y_pred[j*num_data+i]
          and you should group grad and hess in this way as well
    Parameters
    ----------
    func: callable
        Expects a callable with signature ``func(y_true, y_pred)``:

        y_true: array_like of shape [n_samples]
            The target values
        y_pred: array_like of shape [n_samples] or shape[n_samples* n_class]
            The predicted values


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
        grad, hess = func(labels, preds)
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
                for k in range(num_class):
                    for i in range(num_data):
                        idx = k * num_data + i
                        grad[idx] *= weight[i]
                        hess[idx] *= weight[i]
        return grad, hess
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
    ``objective(y_true, y_pred) -> grad, hess``:

    y_true: array_like of shape [n_samples] 
        The target values
    y_pred: array_like of shape [n_samples] or shape[n_samples* n_class]
        The predicted values

    grad: array_like of shape [n_samples] or shape[n_samples* n_class]
        The value of the gradient for each sample point.
    hess: array_like of shape [n_samples] or shape[n_samples* n_class]
        The value of the second derivative for each sample point

    for multi-class task, the y_pred is group by class_id first, then group by row_id
          if you want to get i-th row y_pred in j-th class, the access way is y_pred[j*num_data+i]
          and you should group grad and hess in this way as well
    """

    def __init__(self, num_leaves=31, max_depth=-1, 
                 learning_rate=0.1, n_estimators=10, max_bin=255,
                 silent=True, objective="regression",  
                 nthread=-1, min_split_gain=0, min_child_weight=5, min_child_samples=10,
                 subsample=1, subsample_freq=1, colsample_bytree=1,
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
        """Get parameters"""
        params = super(LGBMModel, self).get_params(deep=deep)
        params['verbose'] = 0 if self.silent else 1
        if self.nthread <= 0:
            params.pop('nthread', None)
        return params

    def fit(self, X, y, eval_set=None, eval_metric=None,
            early_stopping_rounds=None, verbose=True, train_fields=None, valid_fields=None, other_params=None):
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
        train_fields : dict
            other data file in training data. e.g. train_fields['weight'] is weight data
            support fields: weight, group, init_score
        valid_fields : dict
            other data file in training data. e.g. valid_fields[0]['weight'] is weight data for first valid data
            support fields: weight, group, init_score
        other_params: dict
            other parameters
        """
        evals_result = {}
        params = self.get_params()

        if other_params is not None:
            params.update(other_params)

        if callable(self.objective):
            fobj = _point_wise_objective(self.objective)
            params["objective"] = "None"
        else:
            params["objective"] = self.objective
            fobj = None
        if callable(eval_metric):
            feval = eval_metric
        elif is_str(eval_metric) or isinstance(eval_metric, list):
            feval = None
            params.update({'metric': eval_metric})
        else:
            feval = None
        feval = eval_metric if callable(eval_metric) else None

        self._Booster = train(params, (X, y),
                              self.n_estimators, valid_datas=eval_set,
                              early_stopping_rounds=early_stopping_rounds,
                              evals_result=evals_result, fobj=fobj, feval=feval,
                              verbose_eval=verbose, train_fields=train_fields, valid_fields=valid_fields)

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


class LGBMRegressor(LGBMModel, LGBMRegressorBase):
    __doc__ = """Implementation of the scikit-learn API for LightGBM regression.
    """ + '\n'.join(LGBMModel.__doc__.split('\n')[2:])

class LGBMClassifier(LGBMModel, LGBMClassifierBase):
    __doc__ = """Implementation of the scikit-learn API for LightGBM classification.

    """ + '\n'.join(LGBMModel.__doc__.split('\n')[2:])

    def fit(self, X, y, eval_set=None, eval_metric=None,
            early_stopping_rounds=None, verbose=True, 
            train_fields=None, valid_fields=None, other_params=None):

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        if other_params is None:
            other_params = {}
        if self.n_classes_ > 2:
            # Switch to using a multiclass objective in the underlying LGBM instance
            if not callable(self.objective):
                self.objective = "multiclass"
            other_params['num_class'] = self.n_classes_
        else:
            if not callable(self.objective):
                self.objective = "binary"

        self._le = LGBMLabelEncoder().fit(y)
        training_labels = self._le.transform(y)

        if eval_set is not None:
            eval_set = list( (x[0], self._le.transform(x[1])) for x in eval_set )

        super(LGBMClassifier, self).fit(X, training_labels, eval_set, eval_metric, 
            early_stopping_rounds, verbose, train_fields, valid_fields, other_params)
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
        class_probs = self.booster().predict(data,
                                             raw_score=raw_score,
                                             num_iteration=num_iteration)
        if self.n_classes_ > 2:
            return class_probs
        else:
            classone_probs = class_probs
            classzero_probs = 1.0 - classone_probs
            return np.vstack((classzero_probs, classone_probs)).transpose()

