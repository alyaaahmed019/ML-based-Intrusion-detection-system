"""
Run a trained `ids_bundle.pkl` against a new Wireshark CSV capture. Mirrors
the training-time pipeline exactly: load & clean -> per-packet features ->
sliding-window aggregation -> scale (if needed) -> predict.

Two entry points:
    predict_csv     — uses whichever model in the bundle scored best (F1)
    predict_csv_rf  — always uses the Random Forest member (no scaling needed,
                       useful as a fast, dependency-light default)

CLI usage:
    python -m ids_pipeline.predict path/to/capture.csv --bundle loads/ids_bundle.pkl
"""

import argparse

import joblib
import numpy as np
import pandas as pd

from . import config, data_loading, features
from .utils import predict_with_decision_bias, predict_with_proba_threshold


def _run_shared_pipeline(csv_path: str, nrows: int | None):
    """Load & clean -> per-packet features -> window features. Shared by
    both predict_csv and predict_csv_rf."""
    df = data_loading.load_and_clean(csv_path, nrows=nrows)
    df = features.extract_per_packet_features(df)
    return df


def predict_csv(csv_path: str, bundle_path: str = str(config.OUTPUT_BUNDLE), nrows: int | None = None) -> pd.DataFrame:
    """
    Full inference pipeline using whichever model the bundle marked as best.

    Returns a DataFrame with columns:
        window_idx, time_start, time_end,
        prediction (class name), confidence,
        proba_<ClassName> (one per class, when available)
    """
    bundle = joblib.load(bundle_path)
    best_name = bundle["best_model"]
    clf = bundle["models"][best_name]
    scaler = bundle["scaler"]
    class_names = bundle["class_names"]
    window_size = bundle.get("window_size", config.WINDOW_SIZE)
    step_size = bundle.get("step_size", config.STEP_SIZE)
    threshold = bundle.get("thresholds", {}).get(best_name, 0.5)
    bias = bundle.get("biases", {}).get(best_name, 0.0)

    clf_name = clf.__class__.__name__
    needs_scaling = any(x in clf_name for x in
                         ["Logistic", "SVC", "Linear", "SGD", "MLP", "KNeighbors", "Voting"])

    df = _run_shared_pipeline(csv_path, nrows)
    features_df = features.extract_window_features(df, window_size, step_size)

    if features_df.empty:
        print(f"[WARN] Not enough packets for even one window (need {window_size}, got {len(df)})")
        return pd.DataFrame()

    X = features_df[config.FEATURE_COLS].fillna(0).values.astype(np.float32)
    X_input = scaler.transform(X) if needs_scaling else X

    probas = None
    if hasattr(clf, "predict_proba"):
        probas = clf.predict_proba(X_input)
        pred_idx = predict_with_proba_threshold(probas, threshold)
        confidence = probas[np.arange(len(pred_idx)), pred_idx]
    else:
        scores = clf.decision_function(X_input)
        pred_idx = predict_with_decision_bias(scores, bias)
        confidence = np.abs(scores).max(axis=1) if scores.ndim > 1 else np.abs(scores)

    out = features_df[["window_idx", "time_start", "time_end"]].copy()
    out["prediction"] = [class_names.get(int(i), f"Class_{i}") for i in pred_idx]
    out["confidence"] = confidence.round(4)

    if probas is not None:
        for i, name in sorted(class_names.items()):
            if i < probas.shape[1]:
                out[f"proba_{name}"] = probas[:, i].round(4)

    return out


def predict_csv_rf(csv_path: str, bundle_path: str = str(config.OUTPUT_BUNDLE), nrows: int | None = None) -> pd.DataFrame:
    """Same pipeline as predict_csv but always uses the Random Forest member
    of the bundle (no scaling required — useful as a lightweight default)."""
    bundle = joblib.load(bundle_path)
    clf = bundle["models"]["Random Forest"]
    class_names = bundle["class_names"]
    window_size = bundle.get("window_size", config.WINDOW_SIZE)
    step_size = bundle.get("step_size", config.STEP_SIZE)
    threshold = bundle.get("thresholds", {}).get("Random Forest", 0.5)

    df = _run_shared_pipeline(csv_path, nrows)
    features_df = features.extract_window_features(df, window_size, step_size)

    if features_df.empty:
        print(f"[WARN] Not enough packets for even one window (need {window_size}, got {len(df)})")
        return pd.DataFrame()

    X = features_df[config.FEATURE_COLS].fillna(0).values.astype(np.float32)
    probas = clf.predict_proba(X)
    pred_idx = predict_with_proba_threshold(probas, threshold)
    confidence = probas[np.arange(len(pred_idx)), pred_idx]

    out = features_df[["window_idx", "time_start", "time_end"]].copy()
    out["prediction"] = [class_names.get(int(i), f"Class_{i}") for i in pred_idx]
    out["confidence"] = confidence.round(4)
    for i, name in sorted(class_names.items()):
        if i < probas.shape[1]:
            out[f"proba_{name}"] = probas[:, i].round(4)

    return out


def _cli():
    parser = argparse.ArgumentParser(description="Run IDS inference on a Wireshark CSV capture.")
    parser.add_argument("csv_path", help="Path to the Wireshark CSV export")
    parser.add_argument("--bundle", default=str(config.OUTPUT_BUNDLE), help="Path to ids_bundle.pkl")
    parser.add_argument("--nrows", type=int, default=None, help="Optional row limit for quick tests")
    parser.add_argument("--rf-only", action="store_true", help="Use the Random Forest member only (skip best-model selection)")
    args = parser.parse_args()

    fn = predict_csv_rf if args.rf_only else predict_csv
    results = fn(args.csv_path, args.bundle, nrows=args.nrows)
    print(results.head(10))
    print("\nPrediction counts:")
    print(results["prediction"].value_counts())


if __name__ == "__main__":
    _cli()
