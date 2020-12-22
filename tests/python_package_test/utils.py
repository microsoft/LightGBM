# coding: utf-8
from functools import lru_cache

import sklearn.datasets


@lru_cache(maxsize=None)
def load_boston(**kwargs):
    return sklearn.datasets.load_boston(**kwargs)


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
