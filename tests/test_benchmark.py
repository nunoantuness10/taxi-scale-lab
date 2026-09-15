import json

from taxi_scale_lab.benchmark import run_backend
from taxi_scale_lab.data import prepare
from taxi_scale_lab.report import generate_report


def test_pandas_benchmark_and_report(tmp_path):
    prepare(tmp_path / "data", sizes=(100, 200, 300))
    run = tmp_path / "runs" / "small-pandas.json"
    result = run_backend(tmp_path / "data" / "small.parquet", "pandas", run, repeats=1)
    assert result["metadata"]["backend_status"] == "completed"
    assert all(row["seconds"] >= 0 for row in result["measurements"] if row["status"] == "ok")
    output = generate_report(tmp_path / "runs", tmp_path / "report")
    assert output.exists()
    assert "read_construct" not in (tmp_path / "report" / "operation_by_library.csv").read_text()


def test_run_metadata_is_serializable(tmp_path):
    prepare(tmp_path / "data", sizes=(100, 200, 300))
    path = tmp_path / "run.json"
    run_backend(tmp_path / "data" / "small.parquet", "joblib-pandas", path, repeats=1)
    assert json.loads(path.read_text())["metadata"]["joblib_scope"].startswith("parallel")
