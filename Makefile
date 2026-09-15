.PHONY: install check notebooks demo

install:
	python -m pip install -e '.[dask,dev]'

check:
	ruff check .
	pytest -q

notebooks:
	python scripts/build_notebooks.py

demo:
	taxi-lab prepare --output data/demo --sizes 2000 8000 20000
	taxi-lab benchmark --data data/demo --output results/demo --backends pandas dask joblib-pandas
	taxi-lab train --data data/demo/large.parquet --output results/demo/model.json --max-rows 10000
	taxi-lab report --results results/demo --output results/demo/report
