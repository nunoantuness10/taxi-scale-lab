"""Command-line interface for bounded local experiments."""

from __future__ import annotations

import argparse
import cProfile
import json
from pathlib import Path

from .data import download_month, prepare


def main():
    parser = argparse.ArgumentParser(description="Taxi Scale Lab")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    download = sub.add_parser("download")
    download.add_argument("--month", default="2015-01")
    download.add_argument("--output", default="data/raw")
    prep = sub.add_parser("prepare")
    prep.add_argument("--source")
    prep.add_argument("--output", default="data/demo")
    prep.add_argument("--sizes", type=int, nargs=3, default=[2_000, 8_000, 20_000])
    benchmark = sub.add_parser("benchmark")
    benchmark.add_argument("--data", required=True)
    benchmark.add_argument("--output", default="results/demo")
    benchmark.add_argument("--backends", nargs="+", default=["pandas", "dask", "joblib-pandas"])
    benchmark.add_argument("--workers", type=int, default=2)
    benchmark.add_argument("--repeats", type=int, default=3)
    train = sub.add_parser("train")
    train.add_argument("--data", required=True)
    train.add_argument("--output", default="results/model.json")
    train.add_argument("--max-rows", type=int, default=30_000)
    train.add_argument("--workers", type=int, default=2)
    train.add_argument("--stage", choices=["pickup", "completed-trip"], default="pickup")
    train.add_argument("--profile", help="Write a cProfile .prof file")
    report = sub.add_parser("report")
    report.add_argument("--results", default="results/demo")
    report.add_argument("--output", default="results/demo/report")
    args = parser.parse_args()
    if args.command == "doctor":
        from .benchmark import machine

        print(json.dumps(machine(), indent=2))
        print("RAPIDS: NOT RUN (CPU-only project).")
    elif args.command == "download":
        print(download_month(args.month, args.output))
    elif args.command == "prepare":
        print(prepare(args.output, args.source, tuple(args.sizes)))
    elif args.command == "benchmark":
        from .benchmark import run_backend

        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        paths = (
            sorted(Path(args.data).glob("*.parquet"))
            if Path(args.data).is_dir()
            else [Path(args.data)]
        )
        for path in paths:
            for backend_name in args.backends:
                target = output / f"{path.stem}-{backend_name}.json"
                result = run_backend(path, backend_name, target, args.repeats, args.workers)
                print(target, result["metadata"]["backend_status"])
    elif args.command == "train":
        from .modeling import fit_models

        if args.profile:
            Path(args.profile).parent.mkdir(parents=True, exist_ok=True)
            profiler = cProfile.Profile()
            result = profiler.runcall(
                fit_models, args.data, args.output, args.max_rows, args.workers, args.stage
            )
            profiler.dump_stats(args.profile)
        else:
            result = fit_models(args.data, args.output, args.max_rows, args.workers, args.stage)
        print(
            json.dumps(
                {"regression": result["regression"], "classification": result["classification"]},
                indent=2,
            )
        )
    else:
        from .report import generate_report

        print(generate_report(args.results, args.output))


if __name__ == "__main__":
    main()
