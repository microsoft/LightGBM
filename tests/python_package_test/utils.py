# coding: utf-8
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
