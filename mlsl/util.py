import math

import numpy as np
import pandas


def add_bias_column(X):
    """Idempotently prefix a column of ones to ndarray."""
    ones = np.ones(len(X))
    if not np.array_equal(X[:,0], ones):  # Missing a column of 1s
        return np.c_[ones, X]
    else:
        return X


def print_shape(name, X):
    print("{name}: {X.shape}".format(name=name, X=X))


def print_val(name, val):
    print("{name}: {val}".format(name=name, val=val))


def _obs_exp(a, b):
    return "\nObserved: {}\nExpected: {}".format(a, b)


def to_ndarray(X):
    """Convert whatever into a :class: `numpy.ndarray`."""
    if isinstance(X, np.ndarray):
        return X
    if isinstance(X, pandas.DataFrame) or isinstance(X, pandas.Series):
        return X.values
    else:
        raise Exception(
                "Failed conversion to numpy.ndarray "
                "from unsupported type '{}'".format(type(X)))


def assert_shape(a, x):
    assert a.shape == x, _obs_exp(a.shape, x)


def assertClose(a, b, tol):
    assert np.allclose(a.flatten(), b.flatten(), atol=tol), \
        _obs_exp(a, b)


def to_col_vec(x):
    """Converts a row vector to a column vector.

    If it's a 1D array, then the extra dimension is added.
    If it's a matrix, an exception is raised.
    """
    if x.ndim < 2:  # 1D array
        # Wrap in another array and return the tranpose
        return np.array([x]).T
    if x.shape[0] == 1:  # 2D row vector
        return x.T
    raise ValueError("""
        Cannot convert _x_ to column vector.
        Dimensions: {}
        Shape: {}
        x: {}
    """.format(x.ndim, x.shape, x))
