# Syntax comparison

The adapters expose one benchmark contract, while each engine keeps its native execution model.

| Task | pandas | Dask | pandas API on Spark |
|---|---|---|---|
| Read | `pd.read_parquet(path)` | `dd.read_parquet(path)` | `ps.read_parquet(path)` |
| Filter | `df[df.tip_amt.between(1, 5)]` | same expression, then `compute()` | same expression, action triggers execution |
| Cache | already in process memory | `persist()` | `spark.persist()` |
| Group | `df.groupby(key).fare_amt.mean()` | same, then `compute()` | pandas-like expression over Spark |

PySpark uses expressions such as `df.groupBy(key).agg(F.mean("fare_amt"))`. Joblib is
different: this project uses it to read Parquet row groups concurrently, then performs
the operation in pandas. It is labelled `joblib-pandas` so readers do not mistake it for
a dataframe engine.

Modin is run with its Dask execution engine. RAPIDS/cuDF is recorded as `not run` on the
target Apple Silicon laptop because the required NVIDIA CUDA environment is unavailable.

