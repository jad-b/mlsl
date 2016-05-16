import math
from io import StringIO

import numpy as np
import pandas

from mlsl.log import log


def prepare_data_matrix(X):
    """Idempotently installs a bias column and converts to a numpy.ndarray."""
    # numpy's ndarrays seem to be the simplest option for data operations.
    # Pandas DataFrames have some more complicated behaviour when it comes to
    # iteration.
    npX = to_ndarray(X)
    # Add a "bias" column of 1s. Since multiplying by 1 acts as a no-op, we
    # can learn a parameter to describe the "bias" in the data.
    return add_bias_column(npX)


def to_ndarray(X):
    """Convert *whatever* into a :class: `numpy.ndarray`, or die trying."""
    ret = None
    try:
        if isinstance(X, np.ndarray):
            ret = X
        if isinstance(X, pandas.DataFrame) or isinstance(X, pandas.Series):
            ret = X.values
        else:
            raise Exception(
                    "Failed conversion to numpy.ndarray "
                    "from unsupported type '{}'".format(type(X)))
    finally:
        if ret is not None:
            assert isinstance(ret, np.ndarray) # Call me paranoid.
            return ret


def add_bias_column(X):
    """Idempotently prefix a column of ones to ndarray."""
    ones = np.ones(len(X))
    if not np.array_equal(X[:,0], ones):  # Missing a column of 1s
        return np.c_[ones, X]
    else:
        return X


def to_col_vec(x):
    """Converts a row vector to a column vector.

    If it's a 1D array, then the extra dimension is added.
    If it's a matrix, an exception is raised.
    """
    if x.nimd == 2 and x.shape[1] == 1:
        return x  # Already a column vector
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


def feature_normalize(X):
    """Normalize the features of X."""
    mu = X.mean(axis=1)    # Mean of each column (feature)
    sigma = X.std(axis=1)  # Std. dev. of each feature
    Xn = X- mu
    Xn /= sigma
    return Xn, mu, sigma


def assert_shape(a, x):
    assert a.shape == x, _obs_exp(a.shape, x)


def format_metadata(meta):
    """Prepares a dictionary of metadata for printing."""
    sio = StringIO()
    sio.write("\nMetadata\n========\n")
    for k, v in sorted(meta.items()):
        sio.write("{key} = %({key})".format(key=k))
        if isinstance(v, float):
            sio.write(".3f")
        elif isinstance(v, int):
            sio.write("d")
        elif isinstance(v, str):
            sio.write("s")
        else:
            log.warning("Unrecognized type: %s", type(v))
        sio.write('\n')
    return sio.getvalue()


def print_shape(name, X):
    print("{name}: {X.shape}".format(name=name, X=X))


def print_val(name, val):
    print("{name}: {val}".format(name=name, val=val))


def _obs_exp(a, b):
    return "\nObserved: {}\nExpected: {}".format(a, b)
