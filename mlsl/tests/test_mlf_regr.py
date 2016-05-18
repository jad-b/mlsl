"""
Tests that complete the homework found in `Machine Learning :
Regression ml link`_ course.

.. _ml link: https://www.coursera.org/learn/ml-regression/home/welcome
"""
import numpy as np
import pytest

from mlsl import dataset, MLSL_TESTDATA
from mlsl.log import testlog
from mlsl.linreg import LinearRegression


@pytest.yield_fixture(scope='session')
def wk1_data():
    # Import and split the dataset into training and test sets
    tt_data = dataset.load_and_split('mlfoundations/kc_house_data.csv',
                                     root=MLSL_TESTDATA,
                                     features=['sqft_living'],
                                     target='price',
                                     split=.2)
    yield tt_data


def test_wk1(wk1_data):
    """Week 1 quiz questions."""
    data = wk1_data
    model = LinearRegression()
    model.fit(data.X_train, data.y_train, normalize=True, maxiters=1000)
    import pdb
    pdb.set_trace()

    q1_data = np.array([[2650]])
    testlog.info("Predicted price of 2650 sqft. house: $%.2f",
                 model.predict(q1_data))

    cost, gradient = model.cost(data.X_train, data.y_train)
    testlog.info("RSS using sqft. (training): %.3e", cost)

    testlog.info("Estd. sqft. of $800k home: %.2f", .0)

    testlog.info("RSS using sqft. (test): %.2f", .0)

    testlog.info("RSS using # of bedrooms (test): %.2f", .0)
