import math
import sys
import time

import numpy as np
from scipy import linalg
from sklearn.utils import shuffle

from mlsl.log import log, perflog
from mlsl import util


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
        self.weights = weights
        self.accuracy = None

    def fit(self, X, y, fn=None, **kwargs):
        """Fit the model to the data. Learns || updates model parameters."""
        # Allow the user to determine how we learn the model parameters.
        if fn is None:
            fn = self.batch_gradient_descent
        X, y = util.prepare_data_matrix(X), util.to_ndarray(y)
        if y.ndim != 2 or y.shape[1] != 1:  # If y's not a column vector
            y = y.reshape((len(y), 1))
        log.debug("Fitting model to data (%d samples x %d features)",
                  X.shape[0], X.shape[1])

        self.weights, metadata = fn(X, y, **kwargs)
        if metadata:
            log.info(util.format_metadata(metadata), metadata)

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
        """Predict values of y from X, using our model.

        .. math::
            h_{\\theta}(x) = \\theta_{0} + \\theta_{1}x
        """
        start = time.clock()
        h = X.dot(self.weights)
        perflog.info("Predicted %d values in %.3f seconds", X.shape[1],
                     time.clock() - start)
        return h

    def evaluate(self, X, y):
        """Determine accuracy of our model against labeled data (y)."""
        X, y = util.prepare_data_matrix(X), util.to_ndarray(y)
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
        return Xt.dot(y), None

    # Cost Function
    def _least_squares(self, X, y, lambda_=0, **kwargs):
        """Find the cost using the least squares regression.

        .. math::
            h_{\\theta}(x) = \\theta_{0} + \\theta_{1}x
            J(\\theta) =
                \\frac{1}{2m}\sum_{i=1}^m(h_{\\theta}(x^{i}) - y^{i})^{2}
        """
        assert X.ndim == 2, "X should be a matrix"
        assert y.ndim == 2, "y should be a column vector"
        assert y.shape[1] == 1, "y should be a column vector"
        m = len(y)  # Number of samples
        # J = The cost of our current model
        #   = 1/(2m) * sum((y - Xw)^2)
        #   = 1/(2m) * sum((predictions - actual)^2)
        # Our predictions
        predictions = X.dot(self.weights)
        # The error of our predictions, compared to the actual values
        error = predictions - y
        # J = 1/(2m) * sum(error^2)
        J = (1/(2*m)) * error.T.dot(error)
        assert J.shape == (1, 1)
        # dJ = The gradient||vector of partial derivatives
        #    = 1/m * X'(y - Xw)
        #    = 1/m * X'(error)
        dJ = (1/m) * X.T.dot(error)
        assert dJ.shape == self.weights.shape, \
            "dJ should have the same dims as y"
        return np.asscalar(J), dJ

    # Learning function
    def batch_gradient_descent(self, X, y, alpha=1e-3, maxiters=np.inf,
                               tolerance=1e-9, **kwargs):
        """
        .. math::
            \\theta_{j} := \\alpha\\frac{1}{m}\sum_{i=1}^m
                     (h_{\\theta}(x^{(i)})-y^{(i)})x_{j}^{(i)}
        """
        # A general default is to initialize weights to zero.
        if self.weights is None:
            self.weights = np.zeros((X.shape[1], 1))  # A column vector
            log.info("Zero'd %d model parameters", len(self.weights))

        change = sys.float_info.max  # Largest possible Python float
        prevJ, iters, starttime = np.inf, 0, time.perf_counter()
        while change > tolerance and iters < maxiters:  # Check for convergence
            # Find cost and partial derivatives for update
            J, dJ = self.cost(X, y)
            # Update each weight by a "step", its respective partial derivative
            # Using a vector (numpy array) lets us update in parallel.
            # Multiplying by alpha, a fraction, reduces our chance of
            # overstepping and overshooting our target
            self.weights -= alpha * dJ
            change = math.fabs(prevJ - J)
            log.debug("Change in cost: %.3e", change)
            prevJ, iters = J, iters + 1

        log.info("Converged w/ cost %.3e", J)
        return self.weights, {
            'cost': J,  # Final cost
            'iterations': iters,
            'time': time.perf_counter() - starttime
        }

    # Learning function
    def stochastic_gradient_descent(self, X, y, alpha=1e-3, maxiters=np.inf,
                                    tolerance=1e-9, **kwargs):
        # A general default is to initialize weights to zero.
        if self.weights is None:
            self.weights = np.zeros((X.shape[1], 1))  # A column vector
            log.info("Zero'd %d model parameters", len(self.weights))

        # We can consider iteration through a randomly shuffled X a random
        # sampling from X
        X, y = shuffle(X, y)
        iters, starttime, = 0, time.perf_counter()
        change, prevJ = sys.float_info.max, np.inf
        while change > tolerance and iters < maxiters:
            for i, x in enumerate(X):
                # Find cost and partial derivatives for update
                J, dJ = self.cost(
                        x.reshape((1, x.shape[0])),  # => (1xFeatures) row vec
                        y[i].reshape((1, 1))  # => 1x1 matrix|column vector
                        )
                # assert J < prevJ, "Cost is increasing"
                # Update each weight by a "step", its respective partial
                # derivative. Using a vector (numpy array) lets us update in
                # parallel.  Multiplying by alpha, a fraction, reduces our
                # chance of overstepping and thus overshooting our target
                self.weights -= alpha * dJ
                # Check for convergence
                change = math.fabs(prevJ - J)
                log.debug("Change in cost: %.3e", change)
                if change < tolerance or iters >= maxiters:
                    break
                # Prepare for next iteration
                prevJ, iters = J, iters + 1

        log.info("Converged w/ cost %.3e", J)
        return self.weights, {
            'cost': J,
            'iterations': iters,
            'time': time.perf_counter() - starttime
        }
