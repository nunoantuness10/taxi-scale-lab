"""Build comparison tables using only observed measurements."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def generate_report(results, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    rows, runs, models = [], [], []
    for path in sorted(Path(results).glob("*.json")):
        value = json.loads(path.read_text())
        if "measurements" in value:
            metadata = value["metadata"]
            runs.append(
                {
                    "file": path.name,
                    "backend": metadata["backend"],
                    "status": metadata.get("backend_status"),
                    "dataset": metadata.get("dataset"),
                    "reason": metadata.get("reason", ""),
                }
            )
            rows.extend([{**row, "source_file": path.name} for row in value["measurements"]])
        elif "regression" in value:
            models.append(
                {
                    "file": path.name,
                    "dataset": value["dataset"],
                    "stage": value["stage"],
                    "rmse": value["regression"]["rmse"],
                    "mae": value["regression"]["mae"],
                    "macro_f1": value["classification"]["macro_f1"],
                }
            )
    pd.DataFrame(runs).to_csv(destination / "run_status.csv", index=False)
    pd.DataFrame(models).to_csv(destination / "model_scores.csv", index=False)
    frame = pd.DataFrame(rows)
    table = pd.DataFrame()
    if not frame.empty:
        frame.to_csv(destination / "raw_measurements.csv", index=False)
        good = frame[(frame.status == "ok") & (frame.operation != "read_construct")]
        stats = (
            good.groupby(["dataset", "mode", "operation", "backend"])
            .seconds.agg(
                median_seconds="median", min_seconds="min", max_seconds="max", repeats="count"
            )
            .reset_index()
        )
        stats.to_csv(destination / "timing_statistics.csv", index=False)
        table = good.pivot_table(
            index=["dataset", "mode", "operation"],
            columns="backend",
            values="seconds",
            aggfunc="median",
        )
        table.to_csv(destination / "operation_by_library.csv")
    html = (
        "<html><meta charset='utf-8'><title>Taxi Scale Lab</title>"
        "<style>body{font-family:system-ui;margin:40px}"
        "table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:7px}"
        "div{overflow:auto}</style><h1>Taxi Scale Lab</h1>"
        "<p>Measured observations only. Missing cells are not zero. "
        "GPU runs were not performed.</p>"
    )
    html += "<h2>Run status</h2>" + pd.DataFrame(runs).to_html(index=False)
    html += (
        "<h2>Operations by libraries</h2><div>"
        + (
            table.reset_index().to_html(index=False, float_format=lambda value: f"{value:.5f}")
            if not table.empty
            else "No successful timings"
        )
        + "</div>"
    )
    html += "<h2>Fare models</h2>" + pd.DataFrame(models).to_html(index=False) + "</html>"
    output = destination / "index.html"
    output.write_text(html)
    return output
