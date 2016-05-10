import os
from collections import namedtuple

import pandas
import pytest
from sklearn.cross_validation import train_test_split

from mlsl.linreg import LinearRegression
from mlsl import MLSL_DIR


TrainTestSplit = namedtuple("TrainTestSplit", ["X_train", "X_test",
                                               "y_train", "y_test"])
DATASET_DIR = os.path.joni(MLSL_DIR, 'datasets')
Dataset = namedtuple("Dataset", ["data", "relpath", "target", "accuracy"])
# Collection of datasets fit for Linear Regression
linreg_datasets = (
    (None, 'mlfoundations/kc_house_data.csv', 'price', '1')
)


@pytest.yield_fixture(params=linreg_datasets)
def lr_dataset(request):
    """Provide a dataset for consumption by tests."""
    ds = request.param  # Get this iteration's Dataset
    datapath = os.path.join(DATASET_DIR, ds.relpath)
    # Create a new Dataset tuple with the loaded data
    yield ds._replace(data=pandas.read_csv(datapath),
                      relpath=datapath)


def test_linear_regression(dataset):
    model = LinearRegression()
    y = dataset.data.pop(dataset.target)
    X = dataset.data
    # Split off 20% of our data for testing
    tt_data = TrainTestSplit(train_test_split(X, y, test_size=.2))
    model.fit(tt_data.X_train, tt_data.y_train)
    accuracy = model.evaluate(tt_data.X_test, tt_data.y_test)
    assert accuracy >= .5, "Less than 50% accurate"
