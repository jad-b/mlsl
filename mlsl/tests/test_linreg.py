import textwrap
import time
from collections import namedtuple

import numpy as np
import pytest
from sklearn.cross_validation import train_test_split

from mlsl import dataset, MLSL_TESTDATA
from mlsl.linreg import LinearRegression
from mlsl.log import testlog, perflog


# Collection of datasets fit for Linear Regression
linreg_datasets = (
    dataset.Dataset(
        None, 'mlfoundations/kc_house_data.csv', '1', 'price',
        ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
         'waterfront', 'view', 'condition', 'grade', 'sqft_above',
         'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
         'lat', 'long', 'sqft_living15', 'sqft_lot15', 'price']),
)


@pytest.yield_fixture(params=linreg_datasets)
def dataset(request):
    """Provide a dataset for consumption by tests."""
    ds = request.param  # Get this iteration's Dataset
    # Create a new Dataset tuple with the loaded data
    testlog.debug("Reading Pandas DataFrame from %s...", ds.relpath)
    start = time.clock()
    df = dataset.load(ds.relpath, root=MLSL_TESTDATA)
    # Subset data to just our desired features
    df = df[ds.features]
    perflog.info("Read %s to Pandas DataFrame in %.3f seconds",
                 ds.relpath, time.clock() - start)
    testlog.info(textwrap.dedent("""Loaded DataFrame:
    # of Features: (%d)
    Feature labels: %s
    # of Samples: %d
    Memory Usage: %d bytes
    """), df.shape[1], df.columns, df.shape[0], df.memory_usage().sum())

    yield ds._replace(data=df)


def test_dataset_loading(dataset):
    pass


def test_least_squares():
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
        cost, _ = lr._least_squares(tc.X, tc.y)
        assert np.isclose(tc.cost, cost, 1e-3)


@pytest.mark.skip()
def test_regularized_least_squares():
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
        cost, grad = lr._least_squares(tc.X, tc.y, lambda_=tc.lambda_)
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
        LinearRegression.stochastic_gradient_descent
    )
    for minfn in optimizers:
        for tc in testcases:
            lr = LinearRegression()
            lr.weights = tc.theta
            w, meta = minfn(lr, tc.X, tc.y, alpha=tc.alpha, maxiters=tc.iters)
            assert np.allclose(lr.weights, tc.weights, 1e-4)


def test_linear_regression(dataset):
    # Setup
    model = LinearRegression()
    # Split off the target column from the data
    y = dataset.data.pop(dataset.target)
    testlog.debug("Found %d values for target variable '%s'",
                  len(y), dataset.target)
    X = dataset.data
    testlog.debug("X => (%d, %d)", X.shape[0], X.shape[1])

    # Split off 20% of our data for testing
    start = time.clock()
    tt_data = dataset.TrainTestSplit(*train_test_split(X, y, test_size=.2))
    perflog.debug("Created train/test split in %.3f seconds",
                  time.clock() - start)

    # Execute
    model.fit(tt_data.X_train, tt_data.y_train)

    # Assert
    accuracy = model.evaluate(tt_data.X_test, tt_data.y_test)
    assert accuracy >= .5, "Less than 50% accurate"
