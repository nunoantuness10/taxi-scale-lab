"""Equivalent workloads from the original Koalas/Dask benchmark."""

from __future__ import annotations

import numpy as np

OPERATIONS = [
    "count",
    "count_index",
    "mean",
    "std",
    "mean_sum",
    "sum_columns",
    "mean_product",
    "product_columns",
    "value_counts",
    "mean_complex",
    "complex",
    "groupby",
    "join_count",
    "join",
]
GROUPS = {"fare_amt": ["mean", "std"], "tip_amt": ["mean", "std"]}


def angular_expression(frame):
    """Retain the original arithmetic semantics; this is not a distance estimate."""
    a, b = frame.start_lon * np.pi / 180, frame.end_lon * np.pi / 180
    c, d = frame.start_lat * np.pi / 180, frame.end_lat * np.pi / 180
    intermediate = np.sin((b - a) / 2) ** 2 + np.cos(a) * np.cos(b) * np.sin((d - c) / 2) ** 2
    return 2 * np.arctan2(np.sqrt(intermediate), np.sqrt(1 - intermediate))


def operation(frame, name, other):
    if name == "count":
        return len(frame)
    if name == "count_index":
        return len(frame.index)
    if name == "mean":
        return frame.fare_amt.mean()
    if name == "std":
        return frame.fare_amt.std(ddof=1)
    if name in {"mean_sum", "sum_columns"}:
        result = frame.fare_amt + frame.tip_amt
        return result.mean() if name == "mean_sum" else result
    if name in {"mean_product", "product_columns"}:
        result = frame.fare_amt * frame.tip_amt
        return result.mean() if name == "mean_product" else result
    if name == "value_counts":
        return frame.fare_amt.value_counts()
    if name in {"mean_complex", "complex"}:
        result = angular_expression(frame)
        return result.mean() if name == "mean_complex" else result
    if name == "groupby":
        return frame.groupby("passenger_count").agg(GROUPS)
    if name in {"join", "join_count"}:
        joined = frame.merge(other, left_index=True, right_index=True, how="inner")
        return len(joined) if name == "join_count" else joined
    raise ValueError(f"Unknown operation: {name}")
