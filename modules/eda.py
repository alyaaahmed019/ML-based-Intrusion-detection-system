"""
Exploratory-data-analysis plots for the raw packet stream and the windowed
feature set. Kept separate from the core pipeline (data_loading/features/
train/predict) so those modules don't require matplotlib/seaborn to run in
headless or production environments.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import config


def plot_raw_traffic_eda(df_all: pd.DataFrame, save_path: str = config.EDA_RAW_PNG) -> None:
    """Six-panel EDA on the raw (pre-windowing) merged packet stream:
    class distribution, packet rate over time, packet length distribution,
    protocol breakdown, unique source IPs/sec, and IAT distribution."""
    df_all = df_all.copy()
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    # Class distribution
    ax = axes[0, 0]
    counts = df_all["Label"].value_counts().sort_index()
    colors = [config.CLASS_COLORS[k] for k in counts.index]
    bars = ax.bar([config.CLASS_NAMES[k] for k in counts.index], counts.values,
                  color=colors, edgecolor="white", lw=0.5)
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + counts.max() * 0.01,
                f"{v:,}", ha="center", va="bottom", fontsize=8)
    ax.set_title("Packet class distribution")
    ax.set_ylabel("Packets")
    ax.tick_params(axis="x", rotation=30)

    # Packet rate over time
    ax = axes[0, 1]
    df_all["time_bin"] = df_all["global_time"].astype(int)
    for lbl, color in config.CLASS_COLORS.items():
        pps = df_all[df_all["Label"] == lbl].groupby("time_bin").size()
        if len(pps):
            ax.fill_between(pps.index, pps.values, alpha=0.55, color=color, label=config.CLASS_NAMES[lbl])
    ax.set_title("Packet rate over time")
    ax.set_xlabel("Global time (s)")
    ax.set_ylabel("Pkts / s")
    ax.legend(fontsize=7, loc="upper right")

    # Packet length distribution
    ax = axes[0, 2]
    for lbl, color in config.CLASS_COLORS.items():
        data = df_all[df_all["Label"] == lbl]["Length"]
        if len(data):
            ax.hist(data, bins=50, alpha=0.5, color=color, label=config.CLASS_NAMES[lbl], density=True)
    ax.set_title("Packet length distribution")
    ax.set_xlabel("Length (bytes)")
    ax.legend(fontsize=7)

    # Protocol breakdown
    ax = axes[1, 0]
    proto_counts = df_all["Protocol"].value_counts().head(10)
    ax.barh(proto_counts.index, proto_counts.values, color="#378ADD", edgecolor="white")
    ax.set_title("Protocol breakdown (top 10)")
    ax.set_xlabel("Packets")

    # Unique source IPs per second
    ax = axes[1, 1]
    src_div = df_all.groupby("time_bin")["Source"].nunique()
    ax.plot(src_div.index, src_div.values, color="#7F77DD", lw=1)
    ax.set_title("Unique source IPs per second")
    ax.set_xlabel("Global time (s)")
    ax.set_ylabel("Unique sources")

    # IAT distribution (raw, clipped)
    ax = axes[1, 2]
    df_all["IAT_raw"] = df_all["global_time"].diff().fillna(0).clip(0, 0.1)
    for lbl, color in config.CLASS_COLORS.items():
        d = df_all[df_all["Label"] == lbl]["IAT_raw"]
        if len(d):
            ax.hist(d, bins=60, alpha=0.5, color=color, label=config.CLASS_NAMES[lbl], density=True)
    ax.set_title("IAT distribution (clipped 0.1 s)")
    ax.set_xlabel("IAT (s)")
    ax.legend(fontsize=7)

    plt.suptitle("Raw Traffic EDA", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {save_path}")


def plot_feature_eda(features_df: pd.DataFrame, save_path: str = config.EDA_FEATURES_PNG) -> None:
    """Six-panel EDA on the windowed feature set: syn_ratio, pkt_rate,
    n_unique_dports, IAT_min by class, a bootstrap class-distribution check,
    and a correlation heatmap of key features."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    ax = axes[0, 0]
    for label, color in config.CLASS_COLORS.items():
        d = features_df[features_df["Label"] == label]["syn_ratio"]
        if len(d):
            ax.hist(d, bins=40, alpha=0.6, color=color, label=config.CLASS_NAMES[label], density=True)
    ax.set_title("syn_ratio by class"); ax.set_xlabel("syn_ratio"); ax.legend(fontsize=8)

    ax = axes[0, 1]
    classes_present = sorted(features_df["Label"].unique())
    data = [features_df[features_df["Label"] == l]["pkt_rate"].clip(0, 5000).values for l in classes_present]
    labels = [config.CLASS_NAMES.get(l, str(l)) for l in classes_present]
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for patch, color in zip(bp["boxes"], config.CLASS_COLORS.values()):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax.set_title("pkt_rate distribution (clipped 5k)"); ax.set_ylabel("pkts/s")

    ax = axes[0, 2]
    for label, color in config.CLASS_COLORS.items():
        d = features_df[features_df["Label"] == label]["n_unique_dports"]
        if len(d):
            ax.hist(d, bins=30, alpha=0.6, color=color, label=config.CLASS_NAMES[label], density=True)
    ax.set_title("Unique dst ports per window"); ax.set_xlabel("n_unique_dports"); ax.legend(fontsize=8)

    ax = axes[1, 0]
    for label, color in config.CLASS_COLORS.items():
        d = features_df[features_df["Label"] == label]["IAT_min"].clip(0, 0.01)
        if len(d):
            ax.hist(d, bins=40, alpha=0.6, color=color, label=config.CLASS_NAMES[label], density=True)
    ax.set_title("IAT_min by class (clipped 10ms)"); ax.set_xlabel("IAT_min (s)"); ax.legend(fontsize=8)

    ax = axes[1, 1]
    n_boot = 200
    boot_results = []
    for _ in range(n_boot):
        sample = features_df["Label"].sample(frac=0.5, replace=True)
        boot_results.append(sample.value_counts(normalize=True))
    boot_df = pd.DataFrame(boot_results).fillna(0)
    for label, color in config.CLASS_COLORS.items():
        if label in boot_df.columns:
            ax.hist(boot_df[label] * 100, bins=20, alpha=0.6, color=color, label=config.CLASS_NAMES[label])
    ax.set_title(f"Bootstrap class distribution (n={n_boot})")
    ax.set_xlabel("Class %"); ax.set_ylabel("Frequency"); ax.legend(fontsize=8)

    ax = axes[1, 2]
    key_feats = ["syn_ratio", "pkt_rate", "IAT_min", "n_unique_dports", "n_unique_src", "rst_ratio", "ack_ratio", "Label"]
    corr = features_df[key_feats].corr()
    sns.heatmap(corr, ax=ax, annot=True, fmt=".2f", cmap="coolwarm", center=0, annot_kws={"size": 7})
    ax.set_title("Key feature correlation")

    plt.suptitle("Feature EDA", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {save_path}")


def check_label_leakage(features_df: pd.DataFrame, threshold: float = 0.85) -> pd.Series:
    """Print + return features whose correlation with Label exceeds
    `threshold` — a quick sanity check for accidental label leakage."""
    corr_full = features_df[config.FEATURE_COLS + ["Label"]].corr()
    label_corr = corr_full["Label"].drop("Label").abs().sort_values(ascending=False)
    print("Correlation with Label (top 10):")
    print(label_corr.head(10).round(3))
    suspects = label_corr[label_corr > threshold]
    if len(suspects):
        print(f"\n\u26a0  Leakage suspects: {suspects.index.tolist()}")
    else:
        print("\n\u2713  No leakage suspects")
    return suspects
