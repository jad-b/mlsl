import os
import textwrap
import time
from collections import namedtuple

import pandas
import pytest
from sklearn.cross_validation import train_test_split

from mlsl.linreg import LinearRegression
from mlsl.log import testlog, perflog
from mlsl import MLSL_DIR


TrainTestSplit = namedtuple("TrainTestSplit", ["X_train", "X_test",
                                               "y_train", "y_test"])
DATASET_DIR = os.path.join(MLSL_DIR, 'datasets')
Dataset = namedtuple("Dataset",
        ["data",  # Loaded data
         "relpath",   # Path to the source of the data on the filesystem
         "accuracy",  # Prediction accuracy our model should reach
         "target",  # The feature we're trying to predict
         "features"]  # A sub-set of features of the data we're going to train on
        )
# Collection of datasets fit for Linear Regression
linreg_datasets = (
    Dataset(None, 'mlfoundations/kc_house_data.csv', '1', 'price',
            ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
             'waterfront', 'view', 'condition', 'grade', 'sqft_above',
             'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
             'lat', 'long', 'sqft_living15', 'sqft_lot15', 'price']),
)


@pytest.yield_fixture(params=linreg_datasets)
def dataset(request):
    """Provide a dataset for consumption by tests."""
    ds = request.param  # Get this iteration's Dataset
    datapath = os.path.join(DATASET_DIR, ds.relpath)
    # Create a new Dataset tuple with the loaded data
    testlog.debug("Reading Pandas DataFrame from %s...", ds.relpath)
    start = time.clock()
    df = pandas.read_csv(datapath)
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

    yield ds._replace(data=df, relpath=datapath)


def test_dataset_loading(dataset):
    pass


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
    tt_data = TrainTestSplit(*train_test_split(X, y, test_size=.2))
    perflog.debug("Created train/test split in %.3f seconds",
                  time.clock() - start)

    # Execute
    model.fit(tt_data.X_train, tt_data.y_train)

    # Assert
    accuracy = model.evaluate(tt_data.X_test, tt_data.y_test)
    assert accuracy >= .5, "Less than 50% accurate"
