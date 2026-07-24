# SDN Multi-Class Intrusion Detection System

A machine learning pipeline for detecting network intrusions in
Software-Defined Networking (SDN) environments. Raw traffic is captured from
an SDN controller topology, cleaned, labeled by attack type, and aggregated
into sliding-window flow features (packet rate, byte rate, inter-arrival
times, TCP flag ratios, protocol entropy, etc.). Several classifiers —
Random Forest, Decision Tree, Logistic Regression, Linear SVM, and XGBoost —
are trained and compared, along with a soft-voting ensemble, to classify
traffic into five classes: Normal, DDoS, PortScan, Fuzzer, and BruteForce.

## Layout

```
notebooks/
└── multi_attack_slidingWindow_v4.ipynb  
```

## Notebook vs. package

The notebook (`notebooks/multi_attack_slidingWindow_v4.ipynb`) shows the
experimental draft: the raw EDA (traffic-rate plots, class distributions,
correlation heatmaps), the reasoning behind each labeling rule, and the
trial-and-error of comparing models cell by cell with inline output, kept as
a record of how the approach was developed and why certain design choices
were made (e.g. the per-class temporal split, the downsampling strategy).
The `modules/` package takes that same logic and refactors it for
actual use — modular, importable, testable, and runnable from the command
line or another project (`python -m modules.train`, `python -m
modules.predict`) without needing a Jupyter environment at all.

## Setup

```bash
pip install -r requirements.txt
```

Edit the CSV paths and labeling constants in `config.py` to match your own
Wireshark exports (each must have columns `No., Time, Source, Destination,
Protocol, Length, Info`).

## Train

Runs the full pipeline: load → label → merge → downsample → per-packet
features → sliding-window aggregation → train/test split → train all models
→ soft-voting ensemble → save `loads/ids_bundle.pkl` and `loads/features_final.csv`.

```bash
python -m modules.train
```

To skip the EDA plots (faster, no matplotlib window pop-ups in CI):

``` python
from modules.train import run_pipeline
run_pipeline(run_eda=False)
```

## Predict

```bash
python -m modules.predict path/to/new_capture.csv --bundle loads/ids_bundle.pkl
```

Add `--rf-only` to always use the Random Forest model (no scaling required —
useful as a fast, dependency-light default) instead of whichever model
scored best during training.

Programmatic use:

```python
from modules.predict import predict_csv, predict_csv_rf

results = predict_csv("capture.csv", "loads/ids_bundle.pkl", nrows=1_000_000)
print(results["prediction"].value_counts())
```

## Classes

| Label | Class      | Description                              |
|-------|------------|-------------------------------------------|
| 0     | Normal     | Legitimate background traffic             |
| 1     | DDoS       | SYN flood (TCP)                           |
| 2     | PortScan   | Sequential port probing                   |
| 3     | Fuzzer     | Random / malformed packet injection       |
| 4     | BruteForce | Repeated auth attempts (SSH/FTP/HTTP)     |

## Notes

- Window aggregation defaults to 100 packets/window, sliding 10 packets at a
  time (90% overlap) — configurable via `WINDOW_SIZE`/`STEP_SIZE` in `config.py`.
- The per-class temporal train/test split (first 70%/last 30% of each
  class's time-ordered rows) avoids future-leaking data into training while
  guaranteeing every class appears in both splits.
- `predict.py` re-derives features with the exact same functions used in
  training (`features.py`), so training/inference can't drift out of sync.
