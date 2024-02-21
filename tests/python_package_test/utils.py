# coding: utf-8
import pickle
from functools import lru_cache
from inspect import getfullargspec

import cloudpickle
import joblib
import numpy as np
import sklearn.datasets
from sklearn.utils import check_random_state

import lightgbm as lgb

SERIALIZERS = ["pickle", "joblib", "cloudpickle"]


@lru_cache(maxsize=None)
def load_breast_cancer(**kwargs):
    return sklearn.datasets.load_breast_cancer(**kwargs)


@lru_cache(maxsize=None)
def load_digits(**kwargs):
    return sklearn.datasets.load_digits(**kwargs)


@lru_cache(maxsize=None)
def load_iris(**kwargs):
    return sklearn.datasets.load_iris(**kwargs)


@lru_cache(maxsize=None)
def load_linnerud(**kwargs):
    return sklearn.datasets.load_linnerud(**kwargs)


def make_ranking(
    n_samples=100, n_features=20, n_informative=5, gmax=2, group=None, random_gs=False, avg_gs=10, random_state=0
):
    """Generate a learning-to-rank dataset - feature vectors grouped together with
    integer-valued graded relevance scores. Replace this with a sklearn.datasets function
    if ranking objective becomes supported in sklearn.datasets module.

    Parameters
    ----------
    n_samples : int, optional (default=100)
        Total number of documents (records) in the dataset.
    n_features : int, optional (default=20)
        Total number of features in the dataset.
    n_informative : int, optional (default=5)
        Number of features that are "informative" for ranking, as they are bias + beta * y
        where bias and beta are standard normal variates. If this is greater than n_features, the dataset will have
        n_features features, all will be informative.
    gmax : int, optional (default=2)
        Maximum graded relevance value for creating relevance/target vector. If you set this to 2, for example, all
        documents in a group will have relevance scores of either 0, 1, or 2.
    group : array-like, optional (default=None)
        1-d array or list of group sizes. When `group` is specified, this overrides n_samples, random_gs, and
        avg_gs by simply creating groups with sizes group[0], ..., group[-1].
    random_gs : bool, optional (default=False)
        True will make group sizes ~ Poisson(avg_gs), False will make group sizes == avg_gs.
    avg_gs : int, optional (default=10)
        Average number of documents (records) in each group.
    random_state : int, optional (default=0)
        Random seed.

    Returns
    -------
    X : 2-d np.ndarray of shape = [n_samples (or np.sum(group)), n_features]
        Input feature matrix for ranking objective.
    y : 1-d np.array of shape = [n_samples (or np.sum(group))]
        Integer-graded relevance scores.
    group_ids : 1-d np.array of shape = [n_samples (or np.sum(group))]
        Array of group ids, each value indicates to which group each record belongs.
    """
    rnd_generator = check_random_state(random_state)

    y_vec, group_id_vec = np.empty((0,), dtype=int), np.empty((0,), dtype=int)
    gid = 0

    # build target, group ID vectors.
    relvalues = range(gmax + 1)

    # build y/target and group-id vectors with user-specified group sizes.
    if group is not None and hasattr(group, "__len__"):
        n_samples = np.sum(group)

        for i, gsize in enumerate(group):
            y_vec = np.concatenate((y_vec, rnd_generator.choice(relvalues, size=gsize, replace=True)))
            group_id_vec = np.concatenate((group_id_vec, [i] * gsize))

    # build y/target and group-id vectors according to n_samples, avg_gs, and random_gs.
    else:
        while len(y_vec) < n_samples:
            gsize = avg_gs if not random_gs else rnd_generator.poisson(avg_gs)

            # groups should contain > 1 element for pairwise learning objective.
            if gsize < 1:
                continue

            y_vec = np.append(y_vec, rnd_generator.choice(relvalues, size=gsize, replace=True))
            group_id_vec = np.append(group_id_vec, [gid] * gsize)
            gid += 1

        y_vec, group_id_vec = y_vec[:n_samples], group_id_vec[:n_samples]

    # build feature data, X. Transform first few into informative features.
    n_informative = max(min(n_features, n_informative), 0)
    X = rnd_generator.uniform(size=(n_samples, n_features))

    for j in range(n_informative):
        bias, coef = rnd_generator.normal(size=2)
        X[:, j] = bias + coef * y_vec

    return X, y_vec, group_id_vec


@lru_cache(maxsize=None)
def make_synthetic_regression(n_samples=100, n_features=4, n_informative=2, random_state=42):
    return sklearn.datasets.make_regression(
        n_samples=n_samples, n_features=n_features, n_informative=n_informative, random_state=random_state
    )


def dummy_obj(preds, train_data):
    return np.ones(preds.shape), np.ones(preds.shape)


def mse_obj(y_pred, dtrain):
    y_true = dtrain.get_label()
    grad = y_pred - y_true
    hess = np.ones(len(grad))
    return grad, hess


def softmax(x):
    row_wise_max = np.max(x, axis=1).reshape(-1, 1)
    exp_x = np.exp(x - row_wise_max)
    return exp_x / np.sum(exp_x, axis=1).reshape(-1, 1)


def logistic_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sklearn_multiclass_custom_objective(y_true, y_pred, weight=None):
    num_rows, num_class = y_pred.shape
    prob = softmax(y_pred)
    grad_update = np.zeros_like(prob)
    grad_update[np.arange(num_rows), y_true.astype(np.int32)] = -1.0
    grad = prob + grad_update
    factor = num_class / (num_class - 1)
    hess = factor * prob * (1 - prob)
    if weight is not None:
        weight2d = weight.reshape(-1, 1)
        grad *= weight2d
        hess *= weight2d
    return grad, hess


def pickle_obj(obj, filepath, serializer):
    if serializer == "pickle":
        with open(filepath, "wb") as f:
            pickle.dump(obj, f)
    elif serializer == "joblib":
        joblib.dump(obj, filepath)
    elif serializer == "cloudpickle":
        with open(filepath, "wb") as f:
            cloudpickle.dump(obj, f)
    else:
        raise ValueError(f"Unrecognized serializer type: {serializer}")


def unpickle_obj(filepath, serializer):
    if serializer == "pickle":
        with open(filepath, "rb") as f:
            return pickle.load(f)
    elif serializer == "joblib":
        return joblib.load(filepath)
    elif serializer == "cloudpickle":
        with open(filepath, "rb") as f:
            return cloudpickle.load(f)
    else:
        raise ValueError(f"Unrecognized serializer type: {serializer}")


def pickle_and_unpickle_object(obj, serializer):
    with lgb.basic._TempFile() as tmp_file:
        pickle_obj(obj=obj, filepath=tmp_file.name, serializer=serializer)
        obj_from_disk = unpickle_obj(filepath=tmp_file.name, serializer=serializer)
    return obj_from_disk  # noqa: RET504


# doing this here, at import time, to ensure it only runs once_per import
# instead of once per assertion
_numpy_testing_supports_strict_kwarg = "strict" in getfullargspec(np.testing.assert_array_equal).kwonlyargs


def np_assert_array_equal(*args, **kwargs):
    """
    np.testing.assert_array_equal() only got the kwarg ``strict`` in June 2022:
    https://github.com/numpy/numpy/pull/21595

    This function is here for testing on older Python (and therefore ``numpy``)
    """
    if not _numpy_testing_supports_strict_kwarg:
        kwargs.pop("strict")
    np.testing.assert_array_equal(*args, **kwargs)
