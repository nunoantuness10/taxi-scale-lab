# Verification record

Verified on 15 September 2026 in a bounded CPU build environment. The intended
local setup remains macOS on an Apple M4 MacBook Air with 16 GB RAM.

- Source: January 2015 NYC TLC Yellow Taxi Parquet
- Source rows: 12,741,035
- Nested samples: 10,000 / 50,000 / 200,000 rows
- Benchmark adapters exercised: pandas, Dask and Joblib+pandas
- Benchmark protocol: one warm-up plus three measured repeats
- Models: three pickup-time runs and one completed-trip comparison
- Validation: chronological holdout and expanding-window cross-validation
- Profiling: `cProfile` run generated successfully
- Four explanatory notebooks generated and validated with `nbformat`
- `ruff check .`: passed
- `pytest -q`: 12 passed

The manifest preserves source and sample hashes. Raw repeats, aggregate tables and
model metrics are stored in `real-results/`.
