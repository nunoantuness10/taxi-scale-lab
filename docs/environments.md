# Optional environments

## Spark and pandas-on-Spark

```bash
brew install openjdk@17
export JAVA_HOME="$(brew --prefix openjdk@17)/libexec/openjdk.jdk/Contents/Home"
export PATH="$JAVA_HOME/bin:$PATH"
python3.12 -m venv .venv-spark
source .venv-spark/bin/activate
python -m pip install -e '.[spark]'
export PYARROW_IGNORE_TIMEZONE=1
taxi-lab benchmark --data data/nyc --output results/spark --backends pyspark pandas-spark --workers 2
```

## Modin on Dask

```bash
python3.12 -m venv .venv-modin
source .venv-modin/bin/activate
python -m pip install -e '.[modin]'
taxi-lab benchmark --data data/nyc --output results/modin --backends modin-dask --workers 2
```

RAPIDS cannot run on the Apple GPU. A future GPU extension must use supported NVIDIA/CUDA hardware, synchronize GPU work before stopping timers, and record host-device transfers. See the [RAPIDS installation requirements](https://docs.rapids.ai/install/).
