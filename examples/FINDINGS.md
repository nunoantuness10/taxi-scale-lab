# Real NYC Taxi findings

The bundled results use the official January 2015 NYC TLC Yellow Taxi file. The
source contained 12,741,035 trips; deterministic nested samples contain 10,000,
50,000, and 200,000 rows. These are single-machine CPU measurements, not a cloud
cluster reproduction.

## Dataframe benchmark

Pandas, Dask and Joblib+pandas completed the correctness-validated operations at
all three sample sizes. Coordinate arithmetic was skipped because the current
Parquet schema supplies taxi-zone IDs rather than latitude/longitude. Median
timings and every raw repeat are in `real-results/`.

Timings from the build environment demonstrate the experiment and should not be
presented as measurements from the target Mac. Rerun the documented command on
the Mac before making hardware-specific performance claims.

## Fare prediction

| Sample | Maximum rows | RMSE | MAE | Baseline MAE | R2 | Macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| Small | 10,000 | 7.709 | 4.978 | 5.415 | 0.299 | 0.458 |
| Medium | 30,000 | 7.736 | 5.122 | 5.466 | 0.288 | 0.476 |
| Large | 50,000 | 7.800 | 5.086 | 5.460 | 0.298 | 0.471 |

The pickup-time XGBoost model improves MAE over the median baseline, although its
moderate R2 shows that information available at pickup cannot explain all fare
variation. Logistic regression also exceeds the majority-class accuracy baseline,
but macro F1 exposes weaker performance across fare bands.

The completed-trip comparison on 10,000 rows achieves RMSE 3.498, MAE 1.554,
R2 0.856 and macro F1 0.871. This is expected because trip distance and destination
become available after the trip. It is reported separately and never compared as
if it were a pickup-time forecast.

RAPIDS and physical-cluster experiments were not run because the target machine
has no NVIDIA/CUDA GPU and no cloud credits.

