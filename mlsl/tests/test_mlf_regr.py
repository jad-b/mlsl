"""
Tests that complete the homework found in `Machine Learning :
Regression ml link`_ course.

.. _ml link: https://www.coursera.org/learn/ml-regression/home/welcome
"""
import numpy as np
import pandas
import pytest
from sklearn import linear_model
from sklearn import metrics

from mlsl import dataset, MLSL_TESTDATA
from mlsl.log import testlog
from mlsl.linreg import LinearRegression


@pytest.yield_fixture(scope='session')
def wk1_data():
    # Import and split the dataset into training and test sets
    sqft_data = dataset.load_and_split('mlfoundations/kc_house_data.csv',
                                       root=MLSL_TESTDATA,
                                       features=['sqft_living'],
                                       target='price',
                                       split=.2)
    bedrooms_data = dataset.load_and_split('mlfoundations/kc_house_data.csv',
                                           root=MLSL_TESTDATA,
                                           features=['bedrooms'],
                                           target='price',
                                           split=.2)
    yield sqft_data, bedrooms_data


def evaluation_table(y_true, preds, names):
    """Build a cross-model evaluation report."""
    evals = []
    for p in preds:
        evals.append(evaluate_regr_model(y_true, p))
    return pandas.DataFrame(evals, index=names)


def evaluate_regr_model(y_true, y_pred):
    """Run a series of regression metrics on the predicted values."""
    return {
        'explained_variance': metrics.explained_variance_score(y_true, y_pred),
        'mean_absolute_error': metrics.mean_absolute_error(y_true, y_pred),
        'mean_squared_error': metrics.mean_squared_error(y_true, y_pred),
        'median_absolute_error': metrics.median_absolute_error(y_true, y_pred),
        'r2_score': metrics.r2_score(y_true, y_pred)
    }


def regr_accuracy(predictions, actual, rtol=1e-2, atol=1e-3):
    testlog.info(
        "First three values\npredicted v. actual (error):\n%s",
        '\n'.join(["{:.2f} v. {:.2f} ({:.2f}%)".format(
                  predictions[i][0], actual[i][0],
                  obs_error(predictions[i][0], actual[i][0]))
                  for i in np.random.randint(len(predictions), size=3)]))
    accuracy = np.isclose(predictions, actual, rtol=rtol, atol=atol)
    testlog.info("Accuracy within 1e-3: %.2f",
                 np.count_nonzero(accuracy) / len(accuracy))


def obs_error(obs, exp):
    return ((obs - exp) / exp) * 100.


def test_wk1(wk1_data):
    """Week 1 quiz questions."""
    sqft_data, bedrooms_data = wk1_data
    models = (
        (linear_model.Ridge(alpha=0, fit_intercept=False), 'sklearn.Ridge'),
        (linear_model.LinearRegression(fit_intercept=False), 'sklearn.OLS'),
        (LinearRegression(normalize=True, max_iter=1000), 'mlsl'),
    )
    preds = []
    for clf, name in models:
        data = sqft_data
        clf.fit(data.X_train, data.y_train)
        # Closed-form solution: [-47116.077, 281.96]
        testlog.info("Weights: %s", clf.coef_)
        pred = clf.predict(data.X_train)
        preds.append((clf.predict(data.X_test), name))

        # Homework questions
        q1_data = np.array([[1, 2650]])
        testlog.info("Predicted price of 2650 sqft. house: $%.2f",
                     clf.predict(q1_data))
        # Answer: $700,074.85

        rss = ((pred - data.y_train)**2).sum()
        testlog.info("RSS using sqft. (training): %.3e", rss)
        # Answer: 1.20e15

        # y = mx + b; x = (y-b)/m
        estd_sqft = (8e5 - clf.coef_[0][0]) / clf.coef_[0][1]
        testlog.info("Estd. sqft. of $800k home: %.2f", estd_sqft)
        # Answer: 3004.00

        rss = ((pred - data.y_train)**2).sum()
        testlog.info("RSS using sqft. (test): %.3e", rss)

        data = bedrooms_data
        testlog.info("RSS using # of bedrooms (test): %.2f", .0)
    model_preds, names = list(zip(*preds))
    testlog.info("Evaluation Table\n%s",
                 evaluation_table(sqft_data.y_test, model_preds, names)
                 .to_string())
