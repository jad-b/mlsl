"""
Tests proving certain behaviours of the vendored libraries.
It's not to provde they work, but to provide a reference as to *how*
they work.
"""
import numpy as np


def test_numpy_vector_multiplication():
    # Multiplying a row vector by a column vector produces a 1 x 1 matrix
    result = np.ones((1, 19)).dot(np.ones((19, 1)))
    assert isinstance(result, np.ndarray)
    assert result[0][0] == 19.0  # See? A single scalar inside a 2D array

    # Multiplying a two row vectors produces a scalar
    result = np.ones((19,)).dot(np.ones((19,)))
    assert isinstance(result, np.float64)
    assert result == 19.0

