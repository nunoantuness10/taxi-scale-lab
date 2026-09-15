"""Create executed explanatory notebooks using a small in-process Python executor."""

import ast
import contextlib
import io
import os
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def run(source, namespace, count):
    tree = ast.parse(source)
    last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
    stream = io.StringIO()
    outputs = []
    with contextlib.redirect_stdout(stream):
        exec(compile(tree, "<notebook>", "exec"), namespace)
        if last:
            value = eval(compile(ast.Expression(last.value), "<notebook>", "eval"), namespace)
            if value is not None:
                outputs.append(
                    nbf.v4.new_output(
                        "execute_result", data={"text/plain": repr(value)}, execution_count=count
                    )
                )
    if stream.getvalue():
        outputs.insert(0, nbf.v4.new_output("stream", name="stdout", text=stream.getvalue()))
    return outputs


def build():
    setup = "from taxi_scale_lab.data import synthetic\nimport pandas as pd\ndf = synthetic(2000)\nprint('Synthetic software-check fixture:', len(df), 'rows')"
    specs = [
        (
            "01_data_and_protocol",
            "Data and protocol",
            [
                ("code", setup),
                ("code", "df.head()"),
                (
                    "code",
                    "pd.DataFrame({'dtype': df.dtypes.astype(str), 'missing': df.isna().sum()})",
                ),
            ],
        ),
        (
            "02_operations_and_timings",
            "Operations and timings",
            [
                ("code", setup),
                (
                    "code",
                    "from taxi_scale_lab.benchmark import dimension\nfrom taxi_scale_lab.operations import operation\nlookup = dimension(df)\n{'source_rows': len(df), 'joined_rows': operation(df, 'join_count', lookup), 'filter_retention': len(df[df.tip_amt.between(1,5)]) / len(df)}",
                ),
                ("code", "operation(df[df.tip_amt.between(1,5)], 'groupby', lookup)"),
            ],
        ),
        (
            "03_fare_prediction",
            "Fare prediction",
            [
                ("code", setup),
                (
                    "code",
                    "from taxi_scale_lab.modeling import prepare_ml, fare_classes\ntrain, test, features = prepare_ml(df, 2000)\n{'features': features, 'train_end': str(train.pickup_datetime.max()), 'test_start': str(test.pickup_datetime.min()), 'classes': pd.Series(fare_classes(train.fare_amt)).value_counts().to_dict()}",
                ),
            ],
        ),
        (
            "04_results_and_report",
            "Results and conclusions",
            [
                (
                    "code",
                    "import pandas as pd\nmodels = pd.read_csv('examples/real-results/model_scores.csv')\nmodels",
                ),
                (
                    "code",
                    "timings = pd.read_csv('examples/real-results/timing_statistics.csv')\ntimings.groupby('backend').median_seconds.median().sort_values()",
                ),
                (
                    "code",
                    "{'supported_locally': ['pandas', 'dask', 'joblib-pandas'], 'optional': ['modin-dask', 'pyspark', 'pandas-spark'], 'not_run': ['RAPIDS', 'physical cluster'], 'interpretation': 'timings are machine-specific; missing coordinate operations are skips, not zeros'}",
                ),
            ],
        ),
    ]
    original = Path.cwd()
    os.chdir(ROOT)
    try:
        for filename, title, cells in specs:
            notebook = nbf.v4.new_notebook(
                cells=[
                    nbf.v4.new_markdown_cell(
                        f"# {title}\n\nNuno Antunes · Taxi Scale Lab\n\nOutputs below are executed; synthetic data is explicitly labelled."
                    )
                ]
            )
            namespace = {"__name__": "__main__"}
            for count, (_, source) in enumerate(cells, 1):
                notebook.cells.append(
                    nbf.v4.new_code_cell(
                        source, execution_count=count, outputs=run(source, namespace, count)
                    )
                )
            notebook.metadata.kernelspec = {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            }
            notebook.metadata.language_info = {"name": "python"}
            nbf.validate(notebook)
            nbf.write(notebook, ROOT / "notebooks" / f"{filename}.ipynb")
            print("built", filename)
    finally:
        os.chdir(original)


if __name__ == "__main__":
    build()
