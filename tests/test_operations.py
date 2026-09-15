import numpy as np
import pandas as pd

from taxi_scale_lab.benchmark import assert_equal, dimension
from taxi_scale_lab.data import synthetic
from taxi_scale_lab.operations import OPERATIONS, operation


def test_all_pandas_operations_execute():
    frame = synthetic(500)
    frame.index.name = "row_id"
    lookup = dimension(frame)
    for name in OPERATIONS:
        result = operation(frame, name, lookup)
        assert result is not None


def test_filter_is_inclusive():
    frame = synthetic(100)
    frame.loc[0, "tip_amt"] = 1
    frame.loc[1, "tip_amt"] = 5
    selected = frame[frame.tip_amt.between(1, 5)]
    assert {0, 1}.issubset(selected.index)


def test_correctness_gate():
    assert_equal(pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0]))
    try:
        assert_equal(12, 13)
    except AssertionError:
        pass
    else:
        raise AssertionError("Wrong results must fail validation")


def test_complex_expression_is_finite():
    values = operation(synthetic(100), "complex", pd.DataFrame())
    assert np.isfinite(values).all()
