# coding: utf-8
import sklearn.datasets

try:
    from functools import lru_cache
except ImportError:
    import warnings
    warnings.warn("Could not import functools.lru_cache", RuntimeWarning)

    def lru_cache(maxsize=None):
        cache = {}

        def _lru_wrapper(user_function):
            def wrapper(*args, **kwargs):
                arg_key = (args, tuple(kwargs.items()))
                if arg_key not in cache:
                    cache[arg_key] = user_function(*args, **kwargs)
                return cache[arg_key]
            return wrapper
        return _lru_wrapper


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
