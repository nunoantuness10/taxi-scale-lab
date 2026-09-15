"""Bounded acquisition, deterministic sampling, and explicit provenance."""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ALIASES = {
    "tpep_pickup_datetime": "pickup_datetime",
    "fare_amount": "fare_amt",
    "tip_amount": "tip_amt",
    "trip_distance": "trip_distance",
    "passenger_count": "passenger_count",
    "pickup_longitude": "start_lon",
    "pickup_latitude": "start_lat",
    "dropoff_longitude": "end_lon",
    "dropoff_latitude": "end_lat",
    "PULocationID": "pickup_zone",
    "DOLocationID": "dropoff_zone",
}
CORE = ["fare_amt", "tip_amt", "passenger_count", "trip_distance", "pickup_datetime"]
COORDS = ["start_lon", "start_lat", "end_lon", "end_lat"]


def dump_json(path: str | Path, value: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_month(month: str, directory: str | Path) -> Path:
    if not re.fullmatch(r"20\d{2}-(0[1-9]|1[0-2])", month):
        raise ValueError("Month must use YYYY-MM")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"yellow_tripdata_{month}.parquet"
    if target.exists():
        pq.ParquetFile(target)
        return target
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{target.name}"
    partial = target.with_suffix(".partial")
    if partial.exists():
        raise FileExistsError(f"Move or inspect previous partial download: {partial}")
    limit = 350 * 1024**2
    with urllib.request.urlopen(url, timeout=45) as response, partial.open("xb") as output:
        total = 0
        while block := response.read(1024**2):
            total += len(block)
            if total > limit:
                raise ValueError("Download exceeded the 350 MiB safety limit")
            output.write(block)
    pq.ParquetFile(partial)
    partial.rename(target)
    dump_json(target.with_suffix(".source.json"), {"url": url, "sha256": sha256(target)})
    return target


def synthetic(rows: int = 20_000, seed: int = 42) -> pd.DataFrame:
    """Schema-compatible fixture for software checks, not an NYC data substitute."""
    rng = np.random.default_rng(seed)
    distance = rng.gamma(2, 1.8, rows)
    fare = np.round(3 + 2.6 * distance + rng.gamma(2, 1.2, rows), 2)
    return pd.DataFrame(
        {
            "pickup_datetime": pd.Timestamp("2015-01-01")
            + pd.to_timedelta(rng.integers(0, 31 * 86400, rows), unit="s"),
            "fare_amt": fare,
            "tip_amt": np.round(fare * rng.uniform(0, 0.25, rows), 2),
            "passenger_count": rng.integers(1, 7, rows),
            "trip_distance": distance,
            "start_lon": rng.uniform(-74.05, -73.8, rows),
            "start_lat": rng.uniform(40.65, 40.9, rows),
            "end_lon": rng.uniform(-74.05, -73.8, rows),
            "end_lat": rng.uniform(40.65, 40.9, rows),
        }
    )


def prepare(
    output: str | Path,
    source: str | Path | None = None,
    sizes: tuple[int, int, int] = (2_000, 8_000, 20_000),
    seed: int = 42,
) -> dict[str, object]:
    sizes = tuple(sizes)
    if len(sizes) != 3 or not (100 <= sizes[0] < sizes[1] < sizes[2] <= 500_000):
        raise ValueError("Provide three increasing sizes between 100 and 500000")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    if source:
        parquet = pq.ParquetFile(source)
        available = [column for column in parquet.schema_arrow.names if column in ALIASES]
        pool = pd.DataFrame()
        scanned = 0
        for batch in parquet.iter_batches(batch_size=65_536, columns=available):
            frame = batch.to_pandas().rename(columns=ALIASES)
            missing = set(CORE) - set(frame.columns)
            if missing:
                raise ValueError(f"Missing required fields: {sorted(missing)}")
            frame["sample_key"] = rng.random(len(frame))
            scanned += len(frame)
            pool = pd.concat([pool, frame], ignore_index=True).nsmallest(sizes[-1], "sample_key")
        frame = pool.sort_values("sample_key").drop(columns="sample_key").reset_index(drop=True)
        origin = {
            "kind": "real-tlc",
            "source_file": Path(source).name,
            "source_sha256": sha256(source),
            "source_rows": scanned,
        }
    else:
        frame = synthetic(sizes[-1], seed)
        origin = {"kind": "synthetic", "seed": seed, "source_rows": len(frame)}
    if len(frame) < sizes[-1]:
        raise ValueError("Source has fewer rows than the largest requested sample")
    frame["pickup_datetime"] = pd.to_datetime(frame["pickup_datetime"], errors="coerce")
    for column in frame.columns.difference(["pickup_datetime"]):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    entries = []
    for label, size in zip(["small", "medium", "large"], sizes, strict=True):
        selected = frame.iloc[:size].copy()
        selected.index = pd.Index(np.arange(size), name="row_id")
        path = output / f"{label}.parquet"
        selected.to_parquet(path, row_group_size=25_000)
        entries.append(
            {
                "label": label,
                "rows": size,
                "path": path.name,
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    manifest = {
        "origin": origin,
        "sampling": "nested random-priority samples" if source else "nested synthetic prefixes",
        "seed": seed,
        "coordinates_available": all(column in frame for column in COORDS),
        "datasets": entries,
    }
    dump_json(output / "manifest.json", manifest)
    return manifest
