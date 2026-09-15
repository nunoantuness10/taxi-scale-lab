# Suggested report structure

## 1. Background

Explain pandas, Dask, PySpark, pandas-on-Spark/Koalas, Modin, Joblib, and RAPIDS. Distinguish dataframe libraries from task schedulers and CPU from GPU execution.

## 2. Materials and methods

Report the machine, library versions, worker limits, dataset source and hashes, sample sizes, filtering/caching protocol, validation rules, and ML split. State that samples are nested.

## 3. Original-workload adaptation

Insert `operation_by_library.csv`. Keep standard, filtered, and cached scenarios separate. For speed claims, identify sample, median, and repeat range. Explain lazy materialization and the unusual row-index join.

## 4. Additional libraries

Compare only successful runs. Treat Joblib as parallel ingestion/tuning. Retain Modin fallback warnings. Mark GPU and physical-cluster combinations “not run,” not zero.

## 5. Prediction

Document pickup versus completed-trip features, chronological cross-validation, fixed fare bands, tuned parameters, baselines, and test metrics.

## 6. Discussion and conclusions

Discuss bottlenecks, scheduler overhead, scaling, caching, transfer costs, sample dependence, thermal/background effects, and which operations suit each library. Avoid generalizing local results to a distributed cluster.

Sources: [Databricks benchmark](https://www.databricks.com/blog/2021/04/07/benchmark-koalas-pyspark-and-dask.html), [NYC TLC data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page), [Spark](https://spark.apache.org/docs/3.5.8/), and [RAPIDS](https://docs.rapids.ai/install/).
