"""Backend adapters with explicit materialization boundaries."""

from __future__ import annotations

import os

import pandas as pd

from .operations import GROUPS, operation

BACKENDS = ["pandas", "dask", "joblib-pandas", "modin-dask", "pandas-spark", "pyspark"]


class Backend:
    def __init__(self, name: str, workers: int = 2):
        self.name, self.workers = name, workers
        self.client = self.cluster = self.spark = None
        self.cached = []
        if name == "dask":
            # Some restricted containers omit /proc/meminfo. Dask consults psutil at
            # import time only to choose a CSV block size, although this lab reads
            # Parquet. Supply a conservative value in that unusual environment.
            import psutil

            try:
                psutil.virtual_memory()
            except OSError:
                from types import SimpleNamespace

                psutil.virtual_memory = lambda: SimpleNamespace(total=8 * 1024**3)
            import dask
            import dask.dataframe as dd

            self.config = dask.config.set(scheduler="threads", num_workers=workers)
            self.module = dd
        elif name == "modin-dask":
            os.environ["MODIN_ENGINE"] = "dask"
            os.environ["MODIN_CPUS"] = str(workers)
            import modin.pandas as mpd

            self.module = mpd
        elif name in {"pyspark", "pandas-spark"}:
            os.environ.setdefault("PYARROW_IGNORE_TIMEZONE", "1")
            os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
            from pyspark.sql import SparkSession

            self.spark = (
                SparkSession.builder.master(f"local[{workers}]")
                .appName("taxi-scale-lab")
                .config("spark.ui.enabled", "false")
                .config("spark.driver.memory", "2g")
                .config("spark.sql.shuffle.partitions", str(workers * 2))
                .getOrCreate()
            )
            self.spark.sparkContext.setLogLevel("ERROR")
            if name == "pandas-spark":
                import pyspark.pandas as ps

                self.module = ps
        elif name in {"pandas", "joblib-pandas"}:
            self.module = pd
        else:
            raise ValueError(name)

    def load(self, path):
        if self.name == "pyspark":
            return self.spark.read.parquet(str(path))
        if self.name == "pandas-spark":
            return self.module.read_parquet(str(path), index_col="row_id")
        if self.name == "dask":
            return self.module.read_parquet(str(path), split_row_groups=True)
        if self.name == "joblib-pandas":
            import pyarrow.parquet as pq
            from joblib import Parallel, delayed

            count = pq.ParquetFile(path).num_row_groups

            def read_group(index):
                return pq.ParquetFile(path).read_row_group(index).to_pandas()

            parts = Parallel(n_jobs=self.workers, prefer="threads")(
                delayed(read_group)(index) for index in range(count)
            )
            return pd.concat(parts)
        return self.module.read_parquet(str(path))

    def collect(self, value):
        if self.name == "pyspark" and hasattr(value, "toPandas"):
            result = value.toPandas()
            return result.set_index("row_id") if "row_id" in result else result
        if self.name == "dask" and hasattr(value, "compute"):
            return value.compute()
        if self.name == "modin-dask" and hasattr(value, "_to_pandas"):
            return value._to_pandas()
        if self.name == "pandas-spark" and hasattr(value, "to_pandas"):
            return value.to_pandas()
        return value

    def filter(self, frame):
        return frame[(frame.tip_amt >= 1) & (frame.tip_amt <= 5)]

    def persist(self, frame):
        if self.name == "pyspark":
            frame = frame.cache()
            frame.count()
            self.cached.append(frame)
        elif self.name == "pandas-spark":
            frame = frame.spark.cache()
            len(frame)
            self.cached.append(frame)
        elif self.name == "dask":
            frame = frame.persist()
            self.collect(frame)
        else:
            self.collect(frame)
        return frame

    def dimension(self, pandas_other):
        if self.name == "pyspark":
            return self.spark.createDataFrame(pandas_other.rename_axis("row_id").reset_index())
        if self.name == "dask":
            return self.module.from_pandas(pandas_other, npartitions=1)
        return self.module.DataFrame(pandas_other)

    def execute(self, frame, name, other):
        if self.name == "pyspark":
            return self._spark_execute(frame, name, other)
        result = self.collect(operation(frame, name, other))
        if isinstance(result, pd.Series):
            result.name = None
        return result

    def _spark_execute(self, frame, name, other):
        from pyspark.sql import functions as function

        if name in {"count", "count_index"}:
            return frame.count()
        if name in {"mean", "std"}:
            expression = (
                function.avg("fare_amt") if name == "mean" else function.stddev_samp("fare_amt")
            )
            return frame.select(expression).first()[0]
        if name in {"mean_sum", "sum_columns", "mean_product", "product_columns"}:
            expression = (
                function.col("fare_amt") + function.col("tip_amt")
                if "sum" in name
                else function.col("fare_amt") * function.col("tip_amt")
            )
            if name.startswith("mean"):
                return frame.select(function.avg(expression)).first()[0]
            result = (
                frame.select("row_id", expression.alias("value")).toPandas().set_index("row_id")
            )
            return result["value"].rename(None)
        if name == "value_counts":
            result = frame.groupBy("fare_amt").count().toPandas().set_index("fare_amt")
            return result["count"].rename(None)
        if name == "groupby":
            result = (
                frame.groupBy("passenger_count")
                .agg(
                    function.avg("fare_amt").alias("fare_mean"),
                    function.stddev_samp("fare_amt").alias("fare_std"),
                    function.avg("tip_amt").alias("tip_mean"),
                    function.stddev_samp("tip_amt").alias("tip_std"),
                )
                .toPandas()
                .set_index("passenger_count")
            )
            result.columns = pd.MultiIndex.from_product([list(GROUPS), ["mean", "std"]])
            return result
        if name in {"join", "join_count"}:
            joined = frame.join(function.broadcast(other), on="row_id", how="inner")
            return joined.count() if name == "join_count" else self.collect(joined)
        if name in {"complex", "mean_complex"}:
            a, b = function.radians("start_lon"), function.radians("end_lon")
            c, d = function.radians("start_lat"), function.radians("end_lat")
            intermediate = (
                function.sin((b - a) / 2) ** 2
                + function.cos(a) * function.cos(b) * function.sin((d - c) / 2) ** 2
            )
            expression = 2 * function.atan2(
                function.sqrt(intermediate), function.sqrt(1 - intermediate)
            )
            if name == "mean_complex":
                return frame.select(function.avg(expression)).first()[0]
            result = (
                frame.select("row_id", expression.alias("value")).toPandas().set_index("row_id")
            )
            return result["value"].rename(None)
        raise ValueError(name)

    def release(self):
        for frame in self.cached:
            if self.name == "pandas-spark":
                frame.spark.unpersist()
            else:
                frame.unpersist()
        self.cached.clear()

    def close(self):
        self.release()
        if self.spark:
            self.spark.stop()
        if hasattr(self, "config"):
            self.config.__exit__(None, None, None)
