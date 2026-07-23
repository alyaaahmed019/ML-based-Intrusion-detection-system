"""
Load raw Wireshark CSV captures, clean them, assign attack labels per the
rules in config.py, and merge them into a single time-ordered dataset.
"""

import gc
import numpy as np
import pandas as pd

from . import config


def load_and_clean(path: str, nrows: int | None = None) -> pd.DataFrame:
    """
    Read a Wireshark CSV, coerce types, drop bad rows, sort by Time.

    Expected columns: No., Time, Source, Destination, Protocol, Length, Info
    """
    usecols = ["Time", "Source", "Destination", "Protocol", "Length", "Info"]
    dtypes = {
        "Time": np.float64,
        "Source": "category",
        "Destination": "category",
        "Protocol": "category",
        "Length": np.float32,
        "Info": str,
    }
    df = pd.read_csv(path, nrows=nrows, usecols=usecols, dtype=dtypes)
    print(f"  Original : {df.shape}")

    # Drop rows with no destination (malformed / non-IP summary lines)
    df = df.dropna(subset=["Destination"]).reset_index(drop=True)

    df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
    df["Length"] = pd.to_numeric(df["Length"], errors="coerce").fillna(0).astype(np.int32)
    df["Info"] = df["Info"].astype(str)

    df["Protocol"] = df["Protocol"].astype("category")
    df["Source"] = df["Source"].astype("category")
    df["Destination"] = df["Destination"].astype("category")

    df = df.dropna(subset=["Time", "Length"]).reset_index(drop=True)
    df = df.sort_values("Time").reset_index(drop=True)
    print(f"  Cleaned  : {df.shape}")
    return df


def load_ddos_tcp() -> pd.DataFrame:
    """Load the mixed normal+SYN-flood capture and label rows by the
    IP + time-window rule in config.py."""
    print("Loading DDoS...")
    df = load_and_clean(config.DDOS_TCP_CSV)
    df["Label"] = np.where(
        (df["Source"] == config.DDOS_SRC)
        & (df["Destination"] == config.DDOS_DST)
        & (df["Time"] >= config.DDOS_T_START)
        & (df["Time"] <= config.DDOS_T_END),
        1,  # DDoS-TCP
        0,  # Normal
    )
    print("Label distribution:")
    print(df["Label"].value_counts().rename(config.CLASS_NAMES))
    return df


def load_portscan() -> pd.DataFrame:
    print("Loading PortScan ...")
    df = load_and_clean(config.PORTSCAN_CSV, nrows=config.PORTSCAN_NROWS)
    df["Label"] = 2
    print(f"Rows: {len(df):,}")
    return df


def load_fuzzer() -> pd.DataFrame:
    print("Loading Fuzzer ...")
    df = load_and_clean(config.FUZZER_CSV)
    df["Label"] = 3
    print(f"Rows: {len(df):,}")
    return df


def load_bruteforce() -> pd.DataFrame:
    print("Loading BruteForce ...")
    df = load_and_clean(config.BRUTEFORCE_CSV)
    df["Label"] = 4
    print(f"Rows: {len(df):,}")
    return df


def load_all_labeled() -> dict[str, pd.DataFrame]:
    """Load and label every capture. Returns a dict keyed by source name."""
    return {
        "DDoS-TCP": load_ddos_tcp(),
        "PortScan": load_portscan(),
        "Fuzzer": load_fuzzer(),
        "BruteForce": load_bruteforce(),
    }


def merge_and_build_timeline(labeled: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Tag each slice with its source file, offset each capture's Time onto a
    shared 'global_time' axis (so captures don't overlap), concatenate, and
    sort by global_time.
    """
    dfs = []
    for source_name, df in labeled.items():
        d = df.copy()
        d["source_file"] = source_name
        dfs.append(d)

    offset = 0.0
    for d in dfs:
        d["global_time"] = d["Time"] + offset
        offset = d["global_time"].max() + 1.0

    df_all = pd.concat(dfs, ignore_index=True)
    df_all["source_file"] = df_all["source_file"].astype("category")
    df_all = df_all.sort_values("global_time").reset_index(drop=True)

    del dfs
    gc.collect()

    print("===== MERGED DATASET =====")
    print(f"Total packets: {len(df_all):,}")
    print()
    print("Label distribution:")
    for lbl, cnt in df_all["Label"].value_counts().sort_index().items():
        pct = cnt / len(df_all) * 100
        print(f"  {config.CLASS_NAMES[lbl]:<12} {cnt:>8,}  ({pct:.1f}%)")
    print()
    print("Source file counts:")
    print(df_all["source_file"].value_counts().to_string())

    return df_all


def downsample_dataset(
    df: pd.DataFrame,
    sample_frac: float = config.DOWNSAMPLE_FRAC,
    random_state: int = config.RANDOM_SEED,
    min_keep: int = config.DOWNSAMPLE_MIN_KEEP,
    downsample_labels: list[int] | None = None,
) -> pd.DataFrame:
    """
    Selective downsampling by Label.

    Parameters
    ----------
    df                 : merged raw packet frame
    sample_frac        : fraction to keep from downsampled classes
    random_state       : reproducibility seed
    min_keep           : minimum rows guaranteed per downsampled class
    downsample_labels  : list of label indices to downsample.
                          If None, downsamples all classes.
    """
    print("===== BEFORE DOWNSAMPLING =====")
    print(f"Total rows: {len(df):,}")
    print(df["Label"].value_counts().sort_index().rename(config.CLASS_NAMES).to_string())
    print()

    if downsample_labels is None:
        downsample_labels = list(df["Label"].unique())

    parts = []
    for lbl, group in df.groupby("Label"):
        n = len(group)
        if lbl in downsample_labels:
            keep = max(int(n * sample_frac), min_keep) if n > min_keep else n
            parts.append(group.sample(n=keep, random_state=random_state))
        else:
            parts.append(group)

    df_out = (
        pd.concat(parts, ignore_index=True)
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )

    print("===== AFTER DOWNSAMPLING =====")
    print(f"Total rows: {len(df_out):,}")
    vc = df_out["Label"].value_counts().sort_index()
    for lbl, cnt in vc.items():
        flag = "\u2713" if lbl in downsample_labels else "\u2014"
        print(f"  {config.CLASS_NAMES[lbl]:<12} {cnt:>7,}  ({cnt/len(df_out)*100:.1f}%)  [{flag}]")
    return df_out
