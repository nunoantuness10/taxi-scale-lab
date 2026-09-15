"""Chronological fare regression and classification with fold-local preprocessing."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor

from .benchmark import machine
from .data import dump_json, sha256

FEATURES = [
    "passenger_count",
    "pickup_hour",
    "pickup_weekday",
    "start_lon",
    "start_lat",
    "pickup_zone",
]


def prepare_ml(frame, max_rows=30_000, stage="pickup"):
    if not 500 <= max_rows <= 100_000:
        raise ValueError("ML row budget must be between 500 and 100000")
    frame = frame.copy()
    frame["pickup_datetime"] = pd.to_datetime(frame.pickup_datetime, errors="coerce")
    clean = frame[frame.fare_amt.between(0.01, 500) & frame.pickup_datetime.notna()].copy()
    if len(clean) > max_rows:
        clean = clean.sample(max_rows, random_state=42)
    clean = clean.sort_values("pickup_datetime", kind="stable")
    clean["pickup_hour"] = clean.pickup_datetime.dt.hour
    clean["pickup_weekday"] = clean.pickup_datetime.dt.dayofweek
    features = [column for column in FEATURES if column in clean]
    if stage == "completed-trip":
        features += [
            column
            for column in ["trip_distance", "end_lon", "end_lat", "dropoff_zone"]
            if column in clean
        ]
    elif stage != "pickup":
        raise ValueError("stage must be pickup or completed-trip")
    boundary = clean.pickup_datetime.iloc[int(len(clean) * 0.8)]
    train, test = clean[clean.pickup_datetime < boundary], clean[clean.pickup_datetime >= boundary]
    if len(train) < 300 or len(test) < 50:
        raise ValueError("Not enough observations for chronological evaluation")
    return train, test, features


def fare_classes(values):
    return np.digitize(np.asarray(values), [10.0, 25.0])


def chronological_folds(train):
    folds = []
    for left, right in TimeSeriesSplit(n_splits=3).split(train):
        cutoff = train.pickup_datetime.iloc[right[0]]
        left = left[(train.pickup_datetime.iloc[left] < cutoff).to_numpy()]
        folds.append((left, right))
    return folds


def preprocessing(features):
    categories = [column for column in features if column.endswith("_zone")]
    numbers = [column for column in features if column not in categories]
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric, numbers), ("categorical", categorical, categories)]
    )


def fit_models(path, destination, max_rows=30_000, workers=2, stage="pickup"):
    path, destination = Path(path), Path(destination)
    if destination.exists():
        raise FileExistsError(f"Results already exist: {destination}")
    started = time.perf_counter()
    frame = pd.read_parquet(path)
    read_seconds = time.perf_counter() - started
    started = time.perf_counter()
    train, test, features = prepare_ml(frame, max_rows, stage)
    folds = chronological_folds(train)
    x_train, x_test = train[features], test[features]
    y_train, y_test = train.fare_amt, test.fare_amt
    class_train, class_test = fare_classes(y_train), fare_classes(y_test)
    prep_seconds = time.perf_counter() - started
    regression = GridSearchCV(
        Pipeline(
            [
                ("preprocess", preprocessing(features)),
                (
                    "model",
                    XGBRegressor(n_estimators=100, tree_method="hist", n_jobs=1, random_state=42),
                ),
            ]
        ),
        {"model__max_depth": [3, 5], "model__learning_rate": [0.05, 0.15]},
        cv=folds,
        scoring="neg_root_mean_squared_error",
        n_jobs=workers,
    )
    classification = GridSearchCV(
        Pipeline(
            [
                ("preprocess", preprocessing(features)),
                ("model", LogisticRegression(max_iter=2_000, random_state=42)),
            ]
        ),
        {"model__C": [0.1, 1, 10]},
        cv=folds,
        scoring="f1_macro",
        n_jobs=workers,
    )
    started = time.perf_counter()
    with threadpool_limits(limits=1), parallel_backend("threading", n_jobs=workers):
        regression.fit(x_train, y_train)
        classification.fit(x_train, class_train)
    tune_seconds = time.perf_counter() - started
    predicted = regression.predict(x_test)
    labels = classification.predict(x_test)
    baseline = DummyRegressor(strategy="median").fit(np.zeros((len(train), 1)), y_train)
    baseline_c = DummyClassifier(strategy="most_frequent").fit(
        np.zeros((len(train), 1)), class_train
    )
    result = {
        "machine": machine(),
        "dataset": path.name,
        "sha256": sha256(path),
        "stage": stage,
        "raw_rows": len(frame),
        "train_rows": len(train),
        "test_rows": len(test),
        "features": features,
        "train_end": str(train.pickup_datetime.max()),
        "test_start": str(test.pickup_datetime.min()),
        "timings": {
            "read_seconds": read_seconds,
            "preprocessing_seconds": prep_seconds,
            "tuning_seconds": tune_seconds,
        },
        "regression": {
            "rmse": float(np.sqrt(mean_squared_error(y_test, predicted))),
            "mae": float(mean_absolute_error(y_test, predicted)),
            "r2": float(r2_score(y_test, predicted)),
            "baseline_mae": float(
                mean_absolute_error(y_test, baseline.predict(np.zeros((len(test), 1))))
            ),
            "best_params": regression.best_params_,
            "cv_rmse": float(-regression.best_score_),
        },
        "classification": {
            "accuracy": float(accuracy_score(class_test, labels)),
            "macro_precision": float(
                precision_score(class_test, labels, average="macro", zero_division=0)
            ),
            "macro_recall": float(
                recall_score(class_test, labels, average="macro", zero_division=0)
            ),
            "macro_f1": float(f1_score(class_test, labels, average="macro", zero_division=0)),
            "baseline_accuracy": float(
                accuracy_score(class_test, baseline_c.predict(np.zeros((len(test), 1))))
            ),
            "best_params": classification.best_params_,
        },
        "validation": "three expanding chronological folds and final 20% time holdout",
    }
    dump_json(destination, result)
    pd.DataFrame(regression.cv_results_).to_csv(
        destination.with_name(destination.stem + "-regression-cv.csv"), index=False
    )
    pd.DataFrame(classification.cv_results_).to_csv(
        destination.with_name(destination.stem + "-classification-cv.csv"), index=False
    )
    return result
