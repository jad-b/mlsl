import os
from collections import namedtuple

import pandas
import numpy as np

from mlsl import dataset, MLSL_TESTDATA


def test_load():
    TC = namedtuple('TC', ['path', 'root'])
    testcases = (
        TC(os.path.join(MLSL_TESTDATA, 'mlfoundations/kc_house_data.csv'), ''),
        TC('mlfoundations/kc_house_data.csv', MLSL_TESTDATA)
    )
    for tc in testcases:
        df = dataset.load(tc.path, root=tc.root)
        assert isinstance(df, pandas.DataFrame)


def test_load_and_split():
    TC = namedtuple('TC', ['path', 'root', 'features', 'target', 'split'])
    testcases = (
        TC(os.path.join(MLSL_TESTDATA, 'mlfoundations/kc_house_data.csv'), '',
            ['sqft_living'], 'price', 0.2),
        TC('mlfoundations/kc_house_data.csv', MLSL_TESTDATA,
            ['sqft_living', 'bedrooms'], 'price', 0.3)
    )
    for tc in testcases:
        tt_data = dataset.load_and_split(*tc)
        assert isinstance(tt_data, dataset.TrainTestSplit)
        total_rows = tt_data.X_train.shape[0] + tt_data.X_test.shape[0]
        assert set(tt_data.X_train.columns.values) == set(tc.features), \
            "Requested feature(s) not found"
        assert np.isclose(tt_data.X_train.shape[0],
                          (1 - tc.split) * total_rows, 1e-1), \
            "Requested split % not found"
