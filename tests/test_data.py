import json

import pandas as pd
import pytest

from taxi_scale_lab.data import prepare, synthetic


def test_nested_reproducible_fixtures(tmp_path):
    prepare(tmp_path / "a", sizes=(100, 300, 600))
    prepare(tmp_path / "b", sizes=(100, 300, 600))
    small = pd.read_parquet(tmp_path / "a" / "small.parquet")
    large = pd.read_parquet(tmp_path / "a" / "large.parquet")
    pd.testing.assert_frame_equal(small, large.iloc[:100])
    pd.testing.assert_frame_equal(large, pd.read_parquet(tmp_path / "b" / "large.parquet"))
    assert (
        json.loads((tmp_path / "a" / "manifest.json").read_text())["origin"]["kind"] == "synthetic"
    )


def test_prepare_refuses_overwrite(tmp_path):
    prepare(tmp_path, sizes=(100, 200, 300))
    with pytest.raises(FileExistsError):
        prepare(tmp_path, sizes=(100, 200, 300))


def test_size_guard(tmp_path):
    with pytest.raises(ValueError):
        prepare(tmp_path, sizes=(200, 100, 300))


def test_synthetic_schema():
    frame = synthetic(500)
    assert {"fare_amt", "tip_amt", "pickup_datetime", "trip_distance"}.issubset(frame)
    assert len(frame) == 500
