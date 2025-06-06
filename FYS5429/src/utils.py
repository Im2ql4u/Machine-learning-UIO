# utils.py
from functools import wraps
from Config import PARAMS

def inject_params(func):
    """
    A decorator that injects the default parameter dictionary into functions that expect a 'params' keyword argument.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # If 'params' is not explicitly passed, add the default PARAMS.
        if 'params' not in kwargs:
            kwargs['params'] = PARAMS
        return func(*args, **kwargs)
    return wrapper
