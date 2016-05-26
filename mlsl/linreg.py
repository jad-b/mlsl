import math
import sys
import time

import numpy as np
from scipy import linalg
from sklearn.utils import shuffle

from mlsl.log import log, perflog
from mlsl import util


class LinearRegression:
    """Linear regression fits a line to data points.

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

    Evaluation
    ----------

    Regression models can be evaluated by using the Root Mean Squared Error.

    This is the default metric used by `Amazon's ML Regression model`_, against
    which they _also_ compare a baseline model that always predicts the mean.


    .. Amazon's ML Regression model:
        http://docs.aws.amazon.com/machine-learning/latest/dg/regression-model-insights.html
    """

    def __init__(self, *, weights=None, **kwargs):
        self.coef_ = weights
        for k, v in kwargs.items():
            setattr(self, k, v)
        #: Mean and standard deviation used to normalize the data
        self.normal_mean = None
        self.normal_std = None

    @property
    def weights(self):
        return self.coef_

    @weights.setter
    def weights(self, x):
        self.coef_ = x

    @property
    def w(self):
        return self.coef_

    def fit(self, X, y):
        """Fit the model to the data. Learns || updates model parameters."""
        if self.normalize:
            X, mu, sigma = util.feature_normalize(X, mu=self.normal_mean,
                                                  sigma=self.normal_std)
            self.normal_mean = mu
            self.normal_std = sigma

        log.debug("Fitting model to data (%d samples x %d features)",
                  X.shape[0], X.shape[1])

        self.coef_, metadata = self.batch_gradient_descent(X, y)
        if metadata:
            log.info(util.format_metadata(metadata), metadata)

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
        J, dJ = self.least_squares(X, y)
        return J, dJ

    def predict(self, X):
        """Predict values of y from X, using our model.

        .. math::
            h_{\\theta}(x) = \\theta_{0} + \\theta_{1}x
        """
        start = time.clock()
        h = X.dot(self.weights.T)
        perflog.info("Predicted %d values in %.3f seconds", X.shape[1],
                     time.clock() - start)
        return h

    def evaluate(self, X, y):
        """Determine accuracy of our model against labeled data (y)."""
        hypo = X.dot(self.weights)
        assert hypo.shape == y.shape
        rmse = self.rmse(y, hypo)
        # Compare against a baseline model that always predicts the mean
        baseline = np.full(y.shape, y.mean())
        rmse_baseline = self.rmse(y, baseline)
        return rmse, rmse_baseline

    def ols(self, X, y):
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

    # Aliases
    ordinary_least_squares = linear_least_squares = normal_equation = ols

    def rmse(self, y, predictions):
        """
        .. math::

            RMSE = \\sqrt[2]{\\frac{1}{2}\\sum_{i=1}^n(actual - predicted)^2}
            h = self.predict(X)
        """
        error = y - predictions
        squared_error = error.T.dot(error)
        halved_sq_error = squared_error / 2
        _rmse = np.sqrt(halved_sq_error)
        return np.asscalar(_rmse)

    # Cost Function
    def least_squares(self, X, y, lambda_=0, **kwargs):
        """Find the cost using the least squares regression.

        .. math::
            h_{\\theta}(x) = \\theta_{0} + \\theta_{1}x
            J(\\theta) =
                \\frac{1}{2m}\sum_{i=1}^m(h_{\\theta}(x^{i}) - y^{i})^{2}
        """
        m = len(y)  # Number of samples
        # J = The cost of our current model
        #   = 1/(2m) * sum((y - Xw)^2)
        #   = 1/(2m) * sum((predictions - actual)^2)
        # Our predictions
        assert X.shape[1] == self.weights.T.shape[0], \
            ("Can't multiply these matrices: {X.shape} x {w.shape}"
             .format(X=X, w=self.weights))
        predictions = X.dot(self.weights.T)
        # The error of our predictions, compared to the actual values
        error = predictions - y
        # J = 1/(2m) * sum(error^2)
        J = (1/(2*m)) * error.T.dot(error)
        assert J.shape == (1, 1)
        # dJ = The gradient||vector of partial derivatives
        #    = 1/m * X'(y - Xw)
        #    = 1/m * X'(error)
        dJ = (1/m) * X.T.dot(error)
        assert dJ.shape == self.weights.T.shape, \
            "dJ should have the same dims as y"
        return np.asscalar(J), dJ.T

    # Learning function
    def batch_gradient_descent(self, X, y):
        """
        .. math::
            \\theta_{j} := \\alpha\\frac{1}{m}\sum_{i=1}^m
                     (h_{\\theta}(x^{(i)})-y^{(i)})x_{j}^{(i)}
        """
        # A general default is to initialize weights to zero.
        if self.weights is None:
            self.weights = np.zeros((1, X.shape[1]))  # A row vector
            log.info("Zero'd %d model parameters", len(self.weights))

        change = sys.float_info.max  # Largest possible Python float
        prevJ, iters, starttime = np.inf, 0, time.perf_counter()
        last = starttime
        maxiters = self.max_iter or np.inf
        tolerance = getattr(self, 'tolerance', 1e-9)
        alpha = getattr(self, 'alpha', .01)
        while change > tolerance and iters < maxiters:
            # Find cost and partial derivatives for update
            J, dJ = self.cost(X, y)
            # Update each weight by a "step", its respective partial derivative
            # Using a vector (numpy array) lets us update in parallel.
            # Multiplying by alpha, a fraction, reduces our chance of
            # overstepping and overshooting our target
            self.weights -= alpha * dJ
            change = math.fabs(prevJ - J)
            # Throttle output by time
            now = time.perf_counter()
            if (now - last) >= .1:
                log.debug("Change in cost: %.3e", change)
                last = now
            prevJ, iters = J, iters + 1

        log.info("Converged w/ cost %.3e", J)
        return self.weights, {
            'cost': J,  # Final cost
            'iterations': iters,
            'time': time.perf_counter() - starttime
        }

    # Learning function
    def stochastic_gradient_descent(self, X, y):
        # A general default is to initialize weights to zero.
        if self.weights is None:
            self.weights = np.zeros((X.shape[1], 1))  # A column vector
            log.info("Zero'd %d model parameters", len(self.weights))

        # We can consider iteration through a randomly shuffled X a random
        # sampling from X
        X, y = shuffle(X, y)
        iters, starttime, = 0, time.perf_counter()
        change, prevJ = sys.float_info.max, np.inf
        maxiters = self.max_iter or np.inf
        while change > self.tolerance and iters < maxiters:
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
                self.weights -= self.alpha * dJ
                # Check for convergence
                change = math.fabs(prevJ - J)
                log.debug("Change in cost: %.3e", change)
                if change < self.tolerance or iters >= maxiters:
                    break
                # Prepare for next iteration
                prevJ, iters = J, iters + 1

        log.info("Converged w/ cost %.3e", J)
        return self.weights, {
            'cost': J,
            'iterations': iters,
            'time': time.perf_counter() - starttime
        }
