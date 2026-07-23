"""
Train/test split, model zoo (RF, Decision Tree, Logistic Regression, Linear
SVM, optional XGBoost, plus a soft-voting ensemble), per-model threshold
tuning, comparison table, and export of the final `ids_bundle.pkl`.

Run as a script to execute the full pipeline end-to-end:
    python -m ids_pipeline.train
"""

import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

from . import config, data_loading, eda, features
from .utils import predict_with_decision_bias, predict_with_proba_threshold

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not installed — skipping. Install with: pip install xgboost")


def get_model_defs() -> dict:
    """Return {name: (estimator, needs_scaling)} for every model to train."""
    defs = {
        "Random Forest": (
            RandomForestClassifier(n_estimators=100, max_depth=20,
                                    class_weight="balanced", n_jobs=-1, random_state=config.RANDOM_SEED),
            False,
        ),
        "Decision Tree": (
            DecisionTreeClassifier(max_depth=15, class_weight="balanced", random_state=config.RANDOM_SEED),
            False,
        ),
        "Logistic Regression": (
            LogisticRegression(max_iter=500, class_weight="balanced", random_state=config.RANDOM_SEED),
            True,
        ),
        "SVM (Linear)": (
            LinearSVC(max_iter=2000, class_weight="balanced", random_state=config.RANDOM_SEED),
            True,
        ),
    }
    if HAS_XGB:
        defs["XGBoost"] = (
            XGBClassifier(n_estimators=100, max_depth=8, learning_rate=0.1,
                          use_label_encoder=False, eval_metric="mlogloss",
                          random_state=config.RANDOM_SEED, n_jobs=-1),
            False,
        )
    return defs


def temporal_train_test_split(features_df: pd.DataFrame):
    """
    Per-class temporal split: for each class, take the first TRAIN_FRACTION
    of its rows (already time-ordered from feature extraction) as train and
    the rest as test. Preserves temporal ordering within every class while
    guaranteeing all classes appear in both splits.
    """
    X_all = features_df[config.FEATURE_COLS].fillna(0).values.astype(np.float32)
    y_all = features_df["Label"].values.astype(int)

    train_idx, test_idx = [], []
    for cls in np.unique(y_all):
        idx = np.where(y_all == cls)[0]
        split = int(len(idx) * config.TRAIN_FRACTION)
        train_idx.extend(idx[:split])
        test_idx.extend(idx[split:])

    train_idx = np.array(sorted(train_idx))
    test_idx = np.array(sorted(test_idx))

    X_train, X_test = X_all[train_idx], X_all[test_idx]
    y_train, y_test = y_all[train_idx], y_all[test_idx]

    print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    print(f"Train label dist: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"Test  label dist: {dict(zip(*np.unique(y_test,  return_counts=True)))}")
    return X_train, X_test, y_train, y_test


def train_and_evaluate(name, clf, scaled, X_tr, X_te, y_tr, y_te, X_tr_s, X_te_s, classes) -> dict:
    Xtr = X_tr_s if scaled else X_tr
    Xte = X_te_s if scaled else X_te
    n_classes = len(classes)

    fit_params = {}
    if name == "XGBoost":
        fit_params["sample_weight"] = compute_sample_weight(class_weight="balanced", y=y_tr)

    t0 = time.time()
    clf.fit(Xtr, y_tr, **fit_params)
    train_time = time.time() - t0

    y_te_bin = label_binarize(y_te, classes=classes)
    roc_auc = None
    best_t = 0.5
    best_bias = 0.0

    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(Xte)
        if n_classes == 2:
            roc_auc = float(roc_auc_score(y_te_bin, y_prob[:, 1]))
        else:
            roc_auc = float(roc_auc_score(y_te_bin, y_prob, multi_class="ovr", average="macro"))

        best_f1 = -1.0
        for t in np.linspace(0.05, 0.95, 19):
            preds = predict_with_proba_threshold(y_prob, t)
            rep_t = classification_report(y_te, preds, output_dict=True, zero_division=0)
            f1 = rep_t["macro avg"]["f1-score"]
            if f1 > best_f1:
                best_f1, best_t = f1, t
        y_pred = predict_with_proba_threshold(y_prob, best_t)
        print(f"[{name}] Optimal threshold: {best_t:.2f} (macro F1: {best_f1:.4f})")
    elif hasattr(clf, "decision_function"):
        y_score = clf.decision_function(Xte)
        if y_score.ndim == 1:
            y_score = y_score.reshape(-1, 1)
        try:
            roc_auc = float(roc_auc_score(y_te_bin, y_score, multi_class="ovr", average="macro"))
        except Exception:
            roc_auc = None

        best_f1 = -1.0
        for b in np.linspace(-3.0, 3.0, 31):
            preds = predict_with_decision_bias(y_score, b)
            rep_b = classification_report(y_te, preds, output_dict=True, zero_division=0)
            f1 = rep_b["macro avg"]["f1-score"]
            if f1 > best_f1:
                best_f1, best_bias = f1, b
        y_pred = predict_with_decision_bias(y_score, best_bias)
        print(f"[{name}] Optimal bias: {best_bias:.2f} (macro F1: {best_f1:.4f})")
    else:
        y_pred = clf.predict(Xte)

    rep = classification_report(y_te, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_te, y_pred, labels=classes)

    fi = None
    if hasattr(clf, "feature_importances_"):
        fi = dict(zip(config.FEATURE_COLS, clf.feature_importances_.tolist()))
    elif hasattr(clf, "coef_"):
        coef = np.abs(clf.coef_)
        fi = dict(zip(config.FEATURE_COLS, coef.mean(axis=0).tolist()))

    return {
        "clf": clf,
        "train_time": round(train_time, 4),
        "accuracy": round(rep["accuracy"], 4),
        "precision": round(rep["weighted avg"]["precision"], 4),
        "recall": round(rep["weighted avg"]["recall"], 4),
        "f1": round(rep["weighted avg"]["f1-score"], 4),
        "roc_auc": round(roc_auc, 4) if roc_auc else None,
        "cm": cm.tolist(),
        "report": rep,
        "fi": fi,
        "y_pred": y_pred.tolist(),
        "best_t": best_t if hasattr(clf, "predict_proba") else None,
        "best_bias": best_bias if hasattr(clf, "decision_function") else None,
    }


def train_all_models(X_train, X_test, y_train, y_test, X_train_s, X_test_s, classes) -> dict:
    results = {}
    for name, (clf, scaled) in get_model_defs().items():
        print(f"Training {name} ...", end=" ")
        results[name] = train_and_evaluate(
            name, clf, scaled, X_train, X_test, y_train, y_test, X_train_s, X_test_s, classes
        )
        r = results[name]
        auc_str = f"  ROC-AUC={r['roc_auc']}" if r["roc_auc"] else ""
        print(f"acc={r['accuracy']:.4f}  f1={r['f1']:.4f}  t={r['train_time']:.2f}s{auc_str}")
    return results


def add_ensemble(results: dict, X_train_s, X_test_s, y_train, y_test, classes) -> dict:
    """Soft-voting ensemble over Random Forest + Logistic Regression (+
    XGBoost if available). SVM is excluded (no predict_proba); Decision Tree
    is excluded since RF already covers the tree-based signal."""
    rf_est = RandomForestClassifier(n_estimators=100, max_depth=20,
                                     class_weight="balanced", n_jobs=-1, random_state=config.RANDOM_SEED)
    lr_est = LogisticRegression(max_iter=500, class_weight="balanced", random_state=config.RANDOM_SEED)
    estimators = [("rf", rf_est), ("lr", lr_est)]

    if HAS_XGB:
        xgb_est = XGBClassifier(n_estimators=100, max_depth=8, learning_rate=0.1,
                                use_label_encoder=False, eval_metric="mlogloss",
                                random_state=config.RANDOM_SEED, n_jobs=-1)
        estimators.append(("xgb", xgb_est))
        print("Ensemble members: Random Forest + XGBoost + Logistic Regression (soft voting)")
    else:
        print("Ensemble members: Random Forest + Logistic Regression (soft voting) — XGBoost not installed")

    voting_clf = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)

    print("Training Ensemble (Soft Voting) ...", end=" ")
    t0 = time.time()
    voting_clf.fit(X_train_s, y_train)
    train_time_ens = time.time() - t0
    print(f"done in {train_time_ens:.2f}s")

    y_prob_ens = voting_clf.predict_proba(X_test_s)
    y_te_bin = label_binarize(y_test, classes=classes)

    best_t_ens, best_f1_ens = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        preds_ens = predict_with_proba_threshold(y_prob_ens, t)
        rep_t = classification_report(y_test, preds_ens, output_dict=True, zero_division=0)
        f1 = rep_t["macro avg"]["f1-score"]
        if f1 > best_f1_ens:
            best_f1_ens, best_t_ens = f1, t
    y_pred_ens = predict_with_proba_threshold(y_prob_ens, best_t_ens)
    print(f"[Ensemble] Optimal class 0 threshold: {best_t_ens:.2f} (macro F1: {best_f1_ens:.4f})")

    rep_ens = classification_report(y_test, y_pred_ens, output_dict=True, zero_division=0)
    cm_ens = confusion_matrix(y_test, y_pred_ens, labels=classes)
    roc_ens = float(roc_auc_score(y_te_bin, y_prob_ens, multi_class="ovr", average="macro"))

    rf_member = voting_clf.named_estimators_["rf"]
    fi_ens = dict(zip(config.FEATURE_COLS, rf_member.feature_importances_.tolist()))

    results["Ensemble (Soft Voting)"] = {
        "clf": voting_clf,
        "train_time": round(train_time_ens, 4),
        "accuracy": round(rep_ens["accuracy"], 4),
        "precision": round(rep_ens["weighted avg"]["precision"], 4),
        "recall": round(rep_ens["weighted avg"]["recall"], 4),
        "f1": round(rep_ens["weighted avg"]["f1-score"], 4),
        "roc_auc": round(roc_ens, 4),
        "cm": cm_ens.tolist(),
        "report": rep_ens,
        "fi": fi_ens,
        "y_pred": y_pred_ens.tolist(),
        "best_t": best_t_ens,
    }

    r = results["Ensemble (Soft Voting)"]
    print(f"Ensemble -> acc={r['accuracy']:.4f}  f1={r['f1']:.4f}  ROC-AUC={r['roc_auc']:.4f}  t={r['train_time']:.2f}s")
    print(classification_report(y_test, y_pred_ens,
                                 target_names=[config.CLASS_NAMES[c] for c in classes], zero_division=0))
    return results


def build_comparison_table(results: dict) -> pd.DataFrame:
    comparison = pd.DataFrame([
        {"Model": name, "Accuracy": r["accuracy"], "Precision": r["precision"],
         "Recall": r["recall"], "F1-Score": r["f1"],
         "ROC-AUC": r["roc_auc"] if r["roc_auc"] else "N/A",
         "Train Time (s)": r["train_time"]}
        for name, r in results.items()
    ])
    comparison = comparison.sort_values("F1-Score", ascending=False).reset_index(drop=True)
    print(comparison.to_string(index=False))
    return comparison


def save_bundle(results: dict, comparison: pd.DataFrame, scaler: StandardScaler, features_df: pd.DataFrame) -> None:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    best_name = comparison.iloc[0]["Model"]
    print(f"Best model: {best_name}  (F1={comparison.iloc[0]['F1-Score']})")

    features_df.to_csv(config.OUTPUT_FEATURES, index=False)
    print(f"Features saved -> {config.OUTPUT_FEATURES}  ({len(features_df):,} rows)")

    results_meta = {name: {k: v for k, v in r.items() if k != "clf"} for name, r in results.items()}

    bundle = {
        "models": {name: r["clf"] for name, r in results.items()},
        "scaler": scaler,
        "feature_cols": config.FEATURE_COLS,
        "class_names": config.CLASS_NAMES,
        "class_colors": config.CLASS_COLORS,
        "results_meta": results_meta,
        "best_model": best_name,
        "window_size": config.WINDOW_SIZE,
        "step_size": config.STEP_SIZE,
        "thresholds": {name: r.get("best_t", 0.5) for name, r in results.items()},
        "biases": {name: r.get("best_bias", 0.0) for name, r in results.items()},
    }
    joblib.dump(bundle, config.OUTPUT_BUNDLE, compress=3)
    print(f"Bundle saved -> {config.OUTPUT_BUNDLE}")


def run_pipeline(run_eda: bool = True) -> None:
    """End-to-end: load -> merge -> downsample -> extract features -> window
    -> split -> train -> ensemble -> save bundle."""
    labeled = data_loading.load_all_labeled()
    df_all = data_loading.merge_and_build_timeline(labeled)
    df_all = data_loading.downsample_dataset(
        df_all, sample_frac=config.DOWNSAMPLE_FRAC, min_keep=config.DOWNSAMPLE_MIN_KEEP,
        downsample_labels=config.DOWNSAMPLE_LABELS,
    )

    if run_eda:
        eda.plot_raw_traffic_eda(df_all)

    df = features.extract_per_packet_features(df_all)
    print(f"Per-packet features extracted. Shape: {df.shape}")

    print(f"Extracting windows (size={config.WINDOW_SIZE}, step={config.STEP_SIZE}) from {len(df):,} packets ...")
    t0 = time.time()
    features_df = features.extract_window_features(df, config.WINDOW_SIZE, config.STEP_SIZE)
    print(f"Done in {time.time()-t0:.1f}s -> {len(features_df):,} windows")

    if run_eda:
        eda.plot_feature_eda(features_df)
        eda.check_label_leakage(features_df)

    X_train, X_test, y_train, y_test = temporal_train_test_split(features_df)
    classes = sorted(np.unique(features_df["Label"].values.astype(int)))

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = train_all_models(X_train, X_test, y_train, y_test, X_train_s, X_test_s, classes)
    results = add_ensemble(results, X_train_s, X_test_s, y_train, y_test, classes)

    comparison = build_comparison_table(results)
    save_bundle(results, comparison, scaler, features_df)
    print("Done. Run: streamlit run dashboard.py")


if __name__ == "__main__":
    run_pipeline()
