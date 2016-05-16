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

import mlsl.MLSL_DIR


DATASET_DIR = os.path.join(mlsl.MLSL_DIR, 'datasets')
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


def load_and_split(path, root='.', split=0.8):
    """Load and divide a dataset into training/test splits.

    Args:
        path (str): relative path to the dataset
        root (str): Path prefix to search relative to
        split (float): Train/test split, expressed as a decimal.

    Returns:
        TrainTestSplit: A tuple of loaded and split data.
    """
    return TrainTestSplit()
