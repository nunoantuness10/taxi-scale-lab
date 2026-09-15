# Taxi Scale Lab

A reproducible, laptop-sized adaptation of the [2021 Databricks Koalas/Dask benchmark](https://www.databricks.com/blog/2021/04/07/benchmark-koalas-pyspark-and-dask.html), extended with pandas, PySpark, pandas-on-Spark, Modin, Joblib, and a complete taxi-fare ML pipeline.

Designed for a 16 GB Apple M4 Mac. RAPIDS is documented but not executed because it requires supported NVIDIA/CUDA hardware. This is a bounded CPU study, not an exact reproduction of the original 157 GB experiment.

## What it measures

- Standard, filtered, and filtered-plus-cached dataframe operations
- Counts, statistics, arithmetic, value counts, grouping, and joins
- Fully materialized reads in addition to lazy plan construction
- Three nested dataset sizes for scaling analysis
- Pickup-time XGBoost regression for `fare_amount`
- Fixed-band logistic classification with chronological validation
- Optional completed-trip comparison
- Environment versions, dataset hashes, repeat timings, warnings, and failures

Only correctness-validated measurements enter the main result table. Missing cells never mean zero.

## Included real-data evidence

The repository includes a compact, reproducible result snapshot from the January
2015 NYC TLC Yellow Taxi dataset in [`examples/real-results`](examples/real-results).
See [`examples/FINDINGS.md`](examples/FINDINGS.md) for measured outcomes and their
limitations. Raw taxi records are excluded from Git to keep the repository small.

## macOS setup

```bash
brew install python@3.12 libomp
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dask,dev]'
taxi-lab doctor
```

## Quick offline demonstration

```bash
taxi-lab prepare --output data/demo --sizes 2000 8000 20000
taxi-lab benchmark --data data/demo --output results/demo --backends pandas dask joblib-pandas
taxi-lab train --data data/demo/large.parquet --output results/demo/model.json --max-rows 10000
taxi-lab report --results results/demo --output results/demo/report
open results/demo/report/index.html
```

## Real NYC taxi data

Start with a single month and bounded samples:

```bash
taxi-lab download --month 2015-01 --output data/raw
taxi-lab prepare --source data/raw/yellow_tripdata_2015-01.parquet --output data/nyc --sizes 10000 50000 200000
taxi-lab benchmark --data data/nyc --output results/nyc --backends pandas dask joblib-pandas --workers 2 --repeats 3
taxi-lab train --data data/nyc/medium.parquet --output results/nyc/ml-medium.json --max-rows 30000
taxi-lab train --data data/nyc/small.parquet --output results/nyc/ml-profile.json --profile results/nyc/training.prof
taxi-lab report --results results/nyc --output results/nyc/report
```

Inspect the optional profile with `python -m pstats results/nyc/training.prof`.

Current TLC files may contain zone IDs rather than coordinates. Coordinate arithmetic is then explicitly skipped; coordinates are never invented.

## Optional Spark and Modin

Use separate environments as explained in [docs/environments.md](docs/environments.md). Koalas has been replaced by the pandas API on Spark (`pyspark.pandas`). Joblib parallelizes row-group ingestion and model-search tasks; it is not presented as a dataframe engine.

## Notebooks and report

- `01_data_and_protocol.ipynb`
- `02_operations_and_timings.ipynb`
- `03_fare_prediction.ipynb`
- `04_results_and_report.ipynb`

Rebuild executed notebooks with:

```bash
python scripts/build_notebooks.py
```

The report structure is in [docs/report.md](docs/report.md), and assumptions are in [docs/methodology.md](docs/methodology.md).

## Tests

```bash
ruff check .
pytest -q
```

## Author

Nuno Antunes
