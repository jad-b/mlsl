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
import textwrap
from collections import namedtuple

import pandas
from sklearn.cross_validation import train_test_split

from mlsl import util
from mlsl.log import log

# Default to ~/datasets
DATASET_DIR = os.getenv('MLSL_DATASETS',
                        os.path.join(os.path.expanduser('~'), 'datasets'))
TrainTestSplit = namedtuple("TrainTestSplit", ["X_train", "X_test",
                                               "y_train", "y_test"])


def load_and_split(path: str,
                   root='',
                   features=None,
                   target='',
                   split=0.2):
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
    # Extract our "target"; the feature we wish to predict
    y = util.to_col_vec(df.pop(target))
    # Subset the data to just our desired features
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
    fn = None
    if ext == '.csv':
        fn = pandas.read_csv
    if ext == '.json':
        fn = pandas.read_json
    if fn is None:
        raise TypeError("Unsupported file extension: " + ext)
    log.debug("Reading Pandas DataFrame from %s...", path)
    df = fn(path)
    log.info(textwrap.dedent("""Loaded DataFrame:
    # of Features: (%d)
    Feature labels: %s
    # of Samples: %d
    Memory Usage: %d bytes
    """), df.shape[1], df.columns, df.shape[0], df.memory_usage().sum())
    return df
