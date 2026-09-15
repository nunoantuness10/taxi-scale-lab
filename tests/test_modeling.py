import importlib.util

import numpy as np
import pytest

from taxi_scale_lab.data import prepare, synthetic

xgboost_available = importlib.util.find_spec("xgboost") is not None


def test_temporal_split_and_classes():
    from taxi_scale_lab.modeling import chronological_folds, fare_classes, prepare_ml

    train, test, features = prepare_ml(synthetic(1_000), max_rows=1_000)
    assert train.pickup_datetime.max() < test.pickup_datetime.min()
    assert not set(features) & {"fare_amt", "tip_amt", "trip_distance", "end_lon"}
    for left, right in chronological_folds(train):
        assert train.pickup_datetime.iloc[left].max() < train.pickup_datetime.iloc[right].min()
    np.testing.assert_array_equal(fare_classes([9.99, 10, 24.99, 25]), [0, 1, 1, 2])


@pytest.mark.skipif(not xgboost_available, reason="Install core dependencies")
def test_end_to_end_models(tmp_path):
    from taxi_scale_lab.modeling import fit_models

    prepare(tmp_path / "data", sizes=(100, 300, 1_000))
    result = fit_models(tmp_path / "data" / "large.parquet", tmp_path / "model.json", 1_000)
    assert result["regression"]["rmse"] > 0
    assert 0 <= result["classification"]["macro_f1"] <= 1
