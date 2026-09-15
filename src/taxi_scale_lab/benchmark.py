"""Repeated timing with correctness gates and environment capture."""

from __future__ import annotations

import gc
import importlib.metadata
import platform
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from .backends import Backend
from .data import COORDS, dump_json, sha256
from .operations import GROUPS, OPERATIONS, operation

MODES = ["standard", "filtered", "filtered_cached"]


def machine() -> dict[str, object]:
    packages = {}
    for name in [
        "numpy",
        "pandas",
        "pyarrow",
        "dask",
        "modin",
        "pyspark",
        "joblib",
        "scikit-learn",
        "xgboost",
    ]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not installed"
    try:
        memory = round(psutil.virtual_memory().total / 1024**3, 2)
    except (OSError, psutil.Error):
        memory = None
    try:
        logical_cpus = psutil.cpu_count()
    except (OSError, psutil.Error):
        logical_cpus = None
    return {
        "system": platform.system(),
        "architecture": platform.machine(),
        "python": sys.version.split()[0],
        "logical_cpus": logical_cpus,
        "ram_gib": memory,
        "packages": packages,
        "execution": "single-machine CPU",
    }


def assert_equal(actual, expected) -> None:
    if isinstance(expected, pd.DataFrame):
        pd.testing.assert_frame_equal(
            actual.sort_index().sort_index(axis=1),
            expected.sort_index().sort_index(axis=1),
            check_dtype=False,
            check_names=False,
            check_exact=False,
            rtol=1e-5,
            atol=1e-8,
        )
    elif isinstance(expected, pd.Series):
        pd.testing.assert_series_equal(
            actual.sort_index(),
            expected.sort_index(),
            check_dtype=False,
            check_names=False,
            check_exact=False,
            rtol=1e-5,
            atol=1e-8,
        )
    else:
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8, equal_nan=True)


def dimension(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.groupby("passenger_count").agg(GROUPS)
    result.columns = [f"{column}_{stat}" for column, stat in result.columns]
    return result


def run_backend(data_path, backend_name, destination, repeats=3, workers=2):
    path, destination = Path(data_path), Path(destination)
    if destination.exists():
        raise FileExistsError(f"Results already exist: {destination}")
    if repeats < 1 or not 1 <= workers <= 4:
        raise ValueError("Use positive repeats and 1-4 workers")
    import pyarrow.parquet as pq

    if pq.ParquetFile(path).metadata.num_rows > 500_000:
        raise ValueError("Prepare a bounded sample first")
    reference = pd.read_parquet(path)
    metadata = {
        "machine": machine(),
        "backend": backend_name,
        "workers": workers,
        "dataset": path.name,
        "dataset_sha256": sha256(path),
        "rows": len(reference),
        "repeats": repeats,
        "protocol": "uncached end-to-end; cached setup excluded; OS cache not flushed",
        "joblib_scope": "parallel row-group I/O only; pandas global operations",
    }
    records = []
    backend = None
    try:
        started = time.perf_counter()
        backend = Backend(backend_name, workers)
        metadata["startup_seconds"] = time.perf_counter() - started
        for mode in MODES:
            expected_frame = (
                reference if mode == "standard" else reference[reference.tip_amt.between(1, 5)]
            )
            lookup = dimension(expected_frame)
            other = backend.dimension(lookup)
            cached = None
            if mode == "filtered_cached":
                started = time.perf_counter()
                cached = backend.persist(backend.filter(backend.load(path)))
                metadata["cache_setup_seconds"] = time.perf_counter() - started
            names = (
                ["read_construct", "read_collect", *OPERATIONS]
                if mode == "standard"
                else OPERATIONS.copy()
            )
            random.Random(42).shuffle(names)
            for name in names:
                base = {
                    "backend": backend_name,
                    "mode": mode,
                    "operation": name,
                    "dataset": path.stem,
                    "input_rows": len(reference),
                }
                if "complex" in name and not all(column in reference for column in COORDS):
                    records.append(
                        {**base, "status": "skipped", "reason": "source lacks coordinates"}
                    )
                    continue
                expected = (
                    reference
                    if name == "read_collect"
                    else (
                        None
                        if name == "read_construct"
                        else operation(expected_frame, name, lookup)
                    )
                )
                for repeat in range(-1, repeats):
                    gc.collect()
                    with warnings.catch_warnings(record=True) as messages:
                        warnings.simplefilter("always")
                        started = time.perf_counter()
                        try:
                            frame = cached if cached is not None else backend.load(path)
                            if cached is None and mode != "standard":
                                frame = backend.filter(frame)
                            if name == "read_construct":
                                result = None
                            elif name == "read_collect":
                                result = backend.collect(frame)
                            else:
                                result = backend.execute(frame, name, other)
                            seconds = time.perf_counter() - started
                            if name != "read_construct":
                                assert_equal(result, expected)
                            status, reason = "ok", ""
                        except Exception as exc:
                            seconds, status, reason = None, "error", f"{type(exc).__name__}: {exc}"
                    if repeat >= 0 or status == "error":
                        records.append(
                            {
                                **base,
                                "repeat": repeat,
                                "status": status,
                                "seconds": seconds,
                                "reason": reason,
                                "warnings": sorted({str(message.message) for message in messages}),
                                "validated": status == "ok" and name != "read_construct",
                            }
                        )
                    if status == "error":
                        break
            backend.release()
        metadata["backend_status"] = (
            "completed_with_errors"
            if any(row["status"] == "error" for row in records)
            else "completed"
        )
    except Exception as exc:
        metadata["backend_status"] = "unavailable"
        metadata["reason"] = f"{type(exc).__name__}: {exc}"
    finally:
        if backend:
            backend.close()
    result = {"metadata": metadata, "measurements": records}
    dump_json(destination, result)
    return result
