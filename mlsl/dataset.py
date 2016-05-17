"""
dataset
=======
Utilities for loading and saving datasets.

These tools may much more freely leverage existing work in libraries such as
Pandas, Scipy, Numpy, etc. then the actual machine learning algorithms do.
As their concern is getting the data to a consumable state, I don't feel it
necessary to intentionally reinvent the wheel with these guys.
"""
import os
from collections import namedtuple

import pandas
from sklearn.cross_validation import train_test_split


# Default to ~/datasets
DATASET_DIR = os.getenv('MLSL_DATASETS',
                        os.path.join(os.path.expanduser('~'), 'datasets'))
Dataset = namedtuple(
    "Dataset",
    ["data",      # Loaded data
     "relpath",   # Path to the source of the data on the filesystem
     "accuracy",  # Prediction accuracy our model should reach
     "target",    # The feature we're trying to predict
     "features"]  # A sub-set of features of the data we're going to train on
)
TrainTestSplit = namedtuple("TrainTestSplit", ["X_train", "X_test",
                                               "y_train", "y_test"])


def load_and_split(path: str,
                   root='',
                   features=None,
                   target='',
                   split=0.8):
    """Load and divide a dataset into training/test splits.

    Args:
        path (str): relative path to the dataset
        root (str): Path prefix to search relative to
        features (Sequence[str]): List of strings to select as features.
        target (str): Feature to train model to predict.
        split (float): Percentage of data to use as test data, expressed as a
            decimal.

    Returns:
        TrainTestSplit: A tuple of loaded and split data.
    """
    # Load the data
    df = load(path, root=root)
    assert isinstance(df, pandas.DataFrame)
    y = df.pop(target)
    X = df[features]
    # Split the data
    return TrainTestSplit(*train_test_split(X, y, test_size=split))


def load(path, root=''):
    """Load data into a `:class: pandas.DataFrame`.

    Args:
        path (str): relative path to the dataset
        root (str): Path prefix to search relative to. An empty string implies
            'path' is absolute.

    Returns:
        pandas.DataFrame
    """
    if root:
        path = os.path.join(root, path)
    _, ext = os.path.splitext(path)
    if ext == '.csv':
        return pandas.read_csv(path)
    if ext == '.json':
        return pandas.read_json(path)
    raise TypeError("Unsupported file extension: " + ext)
