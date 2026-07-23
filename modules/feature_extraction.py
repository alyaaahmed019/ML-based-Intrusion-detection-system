"""
Per-packet feature extraction (protocol encoding, port/flag parsing, flow IDs,
inter-arrival times) and sliding-window aggregation into fixed-length feature
vectors used for model training and inference.
"""

import gc

import numpy as np
import pandas as pd

from . import config


def extract_per_packet_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add per-packet features to a cleaned packet DataFrame:
      - Protocol_num   : numeric protocol code
      - src_port/dst_port : parsed from the Info column
      - flag_SYN/ACK/FIN/RST/PSH/URG : TCP flag indicators
      - flow_id        : src -> dst : dst_port / proto
      - IAT            : per-flow inter-arrival time

    Drops the 'Info' and 'Protocol' columns to save memory once flags/ports
    have been extracted. Returns the frame sorted back into time order.
    """
    df = df.copy()

    # Protocol encoding
    df["Protocol_num"] = df["Protocol"].astype(str).map(config.PROTOCOL_MAP).fillna(0).astype(np.int16)

    # Port extraction — search (not match) to handle varied Info formats
    ports = df["Info"].str.extract(r"(\d+)\s*>\s*(\d+)")
    df["src_port"] = pd.to_numeric(ports[0], errors="coerce").fillna(-1).astype(np.int32)
    df["dst_port"] = pd.to_numeric(ports[1], errors="coerce").fillna(-1).astype(np.int32)

    # TCP flags
    for flag in ["SYN", "ACK", "FIN", "RST", "PSH", "URG"]:
        df[f"flag_{flag}"] = df["Info"].str.contains(flag, regex=False).astype(np.int8)

    # Drop Info and Protocol early to save memory
    df = df.drop(columns=["Info", "Protocol"])
    gc.collect()

    # Flow ID: src -> dst : dst_port / proto
    df["flow_id"] = (
        df["Source"].astype(str) + "->"
        + df["Destination"].astype(str) + ":"
        + df["dst_port"].astype(str) + "/"
        + df["Protocol_num"].astype(str)
    )
    df["flow_id"] = df["flow_id"].astype("category")

    # Per-flow IAT (MUST sort by flow+time, not global time)
    df = df.sort_values(["flow_id", "Time"]).reset_index(drop=True)
    df["IAT"] = df.groupby("flow_id")["Time"].diff().fillna(0).astype(np.float32)

    # Back to time order for windowing. Training uses 'global_time';
    # standalone inference (no merge step) only has 'Time'.
    sort_col = "global_time" if "global_time" in df.columns else "Time"
    df = df.sort_values(sort_col).reset_index(drop=True)

    return df


def extract_window_features(df: pd.DataFrame, window_size: int, step_size: int) -> pd.DataFrame:
    """Slide a packet-count window over sorted traffic, extract one feature
    vector per window. If df has a 'Label' column, the dominant label per
    window is attached (training mode); otherwise Label/-1 is used
    (inference mode)."""
    n = len(df)
    records = []
    has_labels = "Label" in df.columns

    for start in range(0, n - window_size + 1, step_size):
        end = start + window_size
        w = df.iloc[start:end]
        pkts = len(w)

        # Timing
        t0, t1 = float(w["Time"].iloc[0]), float(w["Time"].iloc[-1])
        dur = max(t1 - t0, 1e-9)
        iats = w["IAT"].values.astype(np.float64)

        # Volume
        lengths = w["Length"].values.astype(np.float64)

        # Flags
        SYN = int(w["flag_SYN"].sum())
        ACK = int(w["flag_ACK"].sum())
        FIN = int(w["flag_FIN"].sum())
        RST = int(w["flag_RST"].sum())
        PSH = int(w["flag_PSH"].sum())
        URG = int(w["flag_URG"].sum())

        # Label: only when the column exists (training mode)
        if has_labels:
            vc = w["Label"].value_counts()
            dominant_label = int(vc.idxmax())
            dominant_ratio = float(vc.max() / pkts)
        else:
            dominant_label = -1  # unknown
            dominant_ratio = -1.0

        # Protocol entropy
        proto_vals = w["Protocol_num"].value_counts(normalize=True).values
        p_entropy = float(-np.sum(proto_vals * np.log2(proto_vals + 1e-9)))

        records.append({
            "window_idx": start,
            "time_start": t0,
            "time_end": t1,
            "pkt_count": pkts,
            "byte_total": float(lengths.sum()),
            "byte_mean": float(lengths.mean()),
            "byte_std": float(lengths.std()) if pkts > 1 else 0.0,
            "byte_min": float(lengths.min()),
            "byte_max": float(lengths.max()),
            "pkt_rate": pkts / dur,
            "byte_rate": float(lengths.sum()) / dur,
            "flow_duration": dur,
            "IAT_mean": float(iats.mean()),
            "IAT_std": float(iats.std()) if pkts > 1 else 0.0,
            "IAT_min": float(iats.min()),
            "IAT_max": float(iats.max()),
            "IAT_total": float(iats.sum()),
            "SYN_count": SYN,
            "ACK_count": ACK,
            "FIN_count": FIN,
            "RST_count": RST,
            "PSH_count": PSH,
            "URG_count": URG,
            "syn_ratio": SYN / pkts,
            "ack_ratio": ACK / pkts,
            "fin_ratio": FIN / pkts,
            "rst_ratio": RST / pkts,
            "psh_ratio": PSH / pkts,
            "syn_no_ack": int(SYN > 0 and ACK == 0),
            "n_unique_src": int(w["Source"].nunique()),
            "n_unique_dst": int(w["Destination"].nunique()),
            "n_unique_flows": int(w["flow_id"].nunique()),
            "n_unique_dports": int(w["dst_port"].nunique()),
            "n_unique_sports": int(w["src_port"].nunique()),
            "protocol_entropy": p_entropy,
            "dominant_ratio": dominant_ratio,
            "Label": dominant_label,
        })

    return pd.DataFrame(records)
