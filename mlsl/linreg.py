import math
import sys
import time

import numpy as np
from scipy import linalg
from sklearn.utils import shuffle

from mlsl.log import log, perflog
from mlsl.util import to_ndarray, add_bias_column


class LinearRegression:
    """Linear regression fits a line to data points, like in a scatter plot.

    You've already encountered linear regression: y = mx + b, or the equation
    for a line, given one input _x_ and two constants, slope (m) and intercept
    (b). A Linear Regression model is interested in learning the slope and
    intercept by slowing tweaking m & b to fit whatever samples of x and y we
    provide. The single biggest difference is that we're now dealing with
    multiple input variables (_x_), and need to learn a _m_ for each.

    Put more formally: The line is represents our predictions of y, along a
    given range of _x_. Now, a line is 2-dimensional, so once we add more than
    one input variable (x), we are no longer dealing with a line, but a plane,
    (for two input variables), and then a hyperplane for any number more than
    two.
    """

    def __init__(self, *, weights=None):
        self.weights = weights or []
        self.accuracy = None

    def fit(self, X, y, fn=None, **kwargs):
        """Fit the model to the data. Learns || updates model parameters."""
        # Allow the user to determine how we learn the model parameters.
        if fn is None:
            fn = self.stochastic_gradient_descent
        # Make sure we're using numpy n-dimensional arrays
        X, y = to_ndarray(X), to_ndarray(y)
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        # Add a "bias" column of 1s. Since multiplying by 1 acts as a no-op, we
        # can learn a parameter to describe the "bias" in the data.
        X = add_bias_column(X)
        log.debug("Fitting model to data (%d samples x %d features)",
                  X.shape[0], X.shape[1])

        start = time.clock()
        self.weights = fn(X, y, **kwargs)
        perflog.info("Fit model in %.3f seconds", time.clock() - start)

        return self.weights

    def cost(self, X, y):
        """A measure of error, between our model and the data.

        "cost" is short for "cost function", whose name lies in the price you
        pay for using a model instead of the One True Model underlying the
        data [LfD, p.38]. Of course, you don't have the One True Model, or
        you'd be using that instead. Hence, machine learning exists!

        :rtype: (float, list[float])
        :return: 2-tuple of cost and gradient.
            The cost represents the overall error of the model in its current
            state, i.e. with the current set of weights.
            The "gradient" is a vector containing the current _magnitude_ of
            error for each input variable. For those familiar with differential
            calculus, it is the partial derivatives of the overall cost with
            respect to each variable.
        """
        J, dJ = self._least_squares(X, y)
        return J, dJ

    def predict(self, X):
        """Predict values of y from X, using our model."""
        start = time.clock()
        h = X.dot(self.weights)
        perflog.info("Predicted %d values in %.3f seconds", X.shape[1],
                     time.clock() - start)
        return h

    def evaluate(self, X, y):
        """Determine accuracy of our model against labeled data (y)."""
        h = self.predict(X)
        assert len(h) == len(y)
        self.accuracy = (h == y).sum() / len(y)
        log.info("Accuracy of %f%% over %d samples", self.accuracy, len(y))
        return self.accuracy

    def _ols(self, X, y):
        """Use Ordinary Least-Squares to fit the Linear Regression weights to
        the data.

        OLS solves finds a global minimum for model coefficients. Said another
        way, it finds the line with the lowest average distance from all data
        points in y.

        See:
            http://docs.scipy.org/doc/scipy/reference/tutorial/linalg.html
                #solving-linear-least-squares-problems-and-pseudo-inverses
        """
        # Find the psuedo-inverse of X
        # (X^T*X)^-1 * X^T
        Xt = linalg.inv(X.T.dot(X)).dot(X.T)
        # Multiply the pseudo-inverse by y
        return Xt.dot(y)

    # Cost Function
    def _least_squares(self, X, y):
        """Find the cost using the least squares regression."""
        # Our predictions
        h = X.dot(self.weights)
        # The error of our predictions, compared to the actual values
        err = y - h
        # The cost of our predictions
        J = err.T.dot(err)
        # The gradient||vector of partial derivatives
        dJ = -2*X.T.dot(h)
        return J, dJ

    def batch_gradient_descent(self, X, y, alpha=.01, tolerance=1e-9, **kwargs):
        change = sys.float_info.max  # Largest possible Python float
        if not self.weights:
            self.weights = np.zeros(X.shape[1])
        prevJ, iters, starttime = .0, 0, time.perf_counter()
        while change > tolerance:  # Check for convergence
            # Find cost and partial derivatives for update
            J, dJ = self.cost(X, y)
            # Update each weight by a "step", its respective partial derivative.
            # Using a vector (numpy array) lets us update in parallel.
            # Multiplying by alpha, a fraction, reduces our chance of
            # overstepping and overshooting our target
            self.weights -= alpha * dJ
            change = math.abs(prevJ - J)
            log.debug("Change in cost: %.3e".format(change))
            prevJ, iters = J, iters + 1

        log.info("Converged w/ cost %.3e".format(J))
        return self.weights, {
            'cost': J,  # Final cost
            'iterations': iters,
            'time': starttime - time.perf_counter()
        }

    def stochastic_gradient_descent(self, X, y, alpha=.01, tolerance=1e-9,
            **kwargs):
        change = sys.float_info.max
        if not self.weights:
            self.weights = np.zeros(X.shape[1])
        prevJ, iters, starttime = .0, 0, time.perf_counter()
        log.info("Shuffling input...")
        # Pandas DataFrame's let you randomly sample a fraction of its values.
        # A fraction of 1 is the same as sampling 100% of all values.
        X = shuffle(X)

        # We can consider iteration through a randomly shuffled X a random
        # sampling from X
        for x in X:
            start = time.clock()
            # Find cost and partial derivatives for update
            J, dJ = self._least_squares(x, y)
            # Update each weight by a "step", its respective partial derivative.
            # Using a vector (numpy array) lets us update in parallel.
            # Multiplying by alpha, a fraction, reduces our chance of
            # overstepping and overshooting our target
            self.weights -= alpha * dJ
            change = math.fabs(prevJ - J)
            log.debug("Change in cost: %.3e", change)
            if change < tolerance:  # Check for convergence
                break
            prevJ, iters = J, iters + 1
            perflog.debug("SGD iteration took %.3f seconds",
                          time.clock() - start)

        log.info("Converged w/ cost %.3e", J)
        return self.weights, {
            'cost': J,
            'iterations': iters,
            'time': starttime - time.perf_counter()
        }
