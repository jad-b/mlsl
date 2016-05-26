from collections import namedtuple

import numpy as np
import pytest

from mlsl import dataset, MLSL_TESTDATA
from mlsl.log import testlog
from mlsl.linreg import LinearRegression
from mlsl.metadata import Metadata


# Collection of datasets fit for Linear Regression
linreg_datasets = (
    Metadata(
        data=None,
        path='mlfoundations/kc_house_data.csv',
        target='price',
        features=[
            'bedrooms', 'bathrooms', 'sqft_living',
            'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
            'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated',
            'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15']
    ),
)


@pytest.yield_fixture(params=linreg_datasets)
def data(request):
    """Provide a dataset for consumption by tests."""
    metadata = request.param  # Get this iteration's Dataset
    # Create a new Dataset tuple with the loaded data
    yield dataset.load_and_split(metadata.path, root=MLSL_TESTDATA,
                                 features=metadata.features,
                                 target=metadata.target)


def testleast_squares():
    TC = namedtuple('TC', ['X', 'y', 'theta', 'cost', 'grad'])
    testcases = [
        TC(np.array([[1, 2], [1, 3], [1, 4], [1, 5]]),
           np.array([[7, 6, 5, 4]]).T,
           np.array([[0.1, 0.2]]).T,
           11.9450,
           None),
        TC(np.array([[1, 2, 3], [1, 3, 4], [1, 4, 5], [1, 5, 6]]),
           np.array([[7, 6, 5, 4]]).T,
           np.array([[0.1, 0.2, 0.3]]).T,
           7.0175,
           None),
    ]
    for tc in testcases:
        lr = LinearRegression()
        lr.weights = tc.theta
        cost, _ = lr.least_squares(tc.X, tc.y)
        assert np.isclose(tc.cost, cost, 1e-3)


@pytest.mark.skip()
def test_regularizedleast_squares():
    TC = namedtuple('TC', ['X', 'y', 'theta', 'lambda_', 'cost', 'grad'])
    testcases = [
        TC(np.array([[1, 1, 1]]),
           np.array([[7, 6, 5]]).T,
           np.array([[.1, .2, .3, .4]]).T,
           0,
           1.3533,
           np.array([[-1.4, -8.7333, -4.3333, -7.9333]]).T),
        TC(np.array([[1, 2, 3, 4]]),
           np.array([[5]].T),
           np.array([[0.1, 0.2, 0.3, 0.4]]).T,
           7,
           3.015,
           np.array([[-2., -2.6, -3.9, -5.2]]).T)
    ]
    for tc in testcases:
        lr = LinearRegression()
        lr.weights = tc.theta
        cost, grad = lr.least_squares(tc.X, tc.y, lambda_=tc.lambda_)
        assert tc.cost == cost
        assert tc.grad == grad


def test_gradient_descent():
    TC = namedtuple('TC', ['X', 'y', 'theta', 'alpha', 'iters', 'weights'])
    testcases = [
        TC(np.array([[1, 5], [1, 2], [1, 4], [1, 5]], dtype=np.int64),
           np.array([[1, 6, 4, 2]]).T,
           np.array([[0., 0.]]).T,
           0.01,
           1000,
           np.array([[5.2148, -0.5733]]).T),
        TC(np.array([[1, 5], [1, 2]]),
           np.array([[1, 6]]).T,
           np.array([[.5, .5]]).T,
           0.1,
           10,
           np.array([[1.70986, 0.19229]]).T)
        ]
    optimizers = (
        LinearRegression.batch_gradient_descent,
        # LinearRegression.stochastic_gradient_descent
    )
    for minfn in optimizers:
        for tc in testcases:
            lr = LinearRegression()
            lr.weights = tc.theta
            w, meta = minfn(lr, tc.X, tc.y, alpha=tc.alpha, maxiters=tc.iters)
            assert np.allclose(lr.weights, tc.weights, 1e-4)


def test_linear_regression(data):
    # Setup
    model = LinearRegression()

    # Execute
    model.fit(data.X_train, data.y_train)

    # Assert
    cost = model.evaluate(data.X_test, data.y_test)
    testlog.info("\nCost of Linear Regression model: %.3e" +
                 "\nCost of Baseline model (mean): %.3e",
                 cost['rss'], cost['baseline'])
    if cost['rss'] > cost['baseline']:
        diff = cost['rss'] / cost['baseline']
        pytest.fail("Our model was {:.2f} costlier than predicting the mean"
                    .format(diff))
