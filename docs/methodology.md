# Methodology

The original benchmark compared Koalas and Dask on 157 GB of 2009–2013 taxi data. This repository independently implements its counts, statistics, arithmetic, value counts, grouping, and row-index joins for bounded local CPU experiments.

Three scenarios are retained: standard, tips filtered inclusively from $1 to $5, and the same filter with engine caching. Uncached timings include input construction/read, filtering where applicable, execution, and output collection. Cached setup occurs before timing and is recorded separately. One warm-up precedes three measured repeats. OS filesystem caches are not flushed.

The original join matches dataframe row indices to an aggregate indexed by passenger count. That unusual semantic is retained with an explicit `row_id`; it is not a general passenger-key join. The original trigonometric expression is retained as an arithmetic workload, not interpreted as geographic distance.

Every result is compared with pandas outside the timed section. Failures and skipped operations remain visible. `read_construct` is excluded from the comparison table because constructing a lazy plan is not a complete read.

Joblib only parallelizes Parquet row groups and independent model fits. Modin uses its Dask engine. Local workers are not a physical cluster. RAPIDS, Dask-RAPIDS, Dask-Modin-RAPIDS, and multi-machine experiments are not run under the agreed CPU-only scope.

Dataset samples are nested and therefore not independent populations. Scheduler overhead at small sizes does not establish large-scale superiority. Compare only matching dataset hashes, machines, worker budgets, versions, and scenarios.
