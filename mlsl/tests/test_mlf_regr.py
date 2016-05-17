"""
Tests that complete the homework found in `Machine Learning :
Regression ml link`_ course.

.. _ml link: https://www.coursera.org/learn/ml-regression/home/welcome
"""
import pytest

import mlsl
from mlsl.log import testlog
from mlsl import dataset


@pytest.yield_fixture(scope='session')
def data():
    # Import and split the dataset into training and test sets
    tt_data = dataset.load_and_split('mlfoundations/kc_house_data.csv',
                                     root=mlsl.MLSL_TESTDATA,
                                     features=['sqft_living'],
                                     target=['price'],
                                     split=.8)
    yield tt_data


def test_wk1(data):
    """Week 1 quiz questions."""
    testlog.info("Predicted price of 2650 sqft. house: %.2f", .0)
    testlog.info("RSS using sqft. (training): %.3f", .0)
    testlog.info("Estd. sqft. of $800k home: %.2f", .0)
    testlog.info("RSS using sqft. (test): %.2f", .0)
    testlog.info("RSS using # of bedrooms (test): %.2f", .0)
