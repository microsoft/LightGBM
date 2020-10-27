
try:
    from functools import lru_cache
except ImportError:
    import warnings
    warnings.warn("Could not import functools.lru_cache", RuntimeWarning)

    def lru_cache(user_function, maxsize=None):
        @wraps(user_function)
        def wrapper(*args, **kwargs):
            arg_key = tuple(args, [item for item in kwargs.items()])
            if arg_key not in cache:
                cache[arg_key] = user_function(*args)
            return cache[arg_key]
        return wrapper
