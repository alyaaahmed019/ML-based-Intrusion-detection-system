"""
Central configuration for the SDN Multi-Class Intrusion Detection pipeline.

Edit the CSV paths below to point at your own Wireshark exports. Each input
file must be a Wireshark CSV export with columns:
    No., Time, Source, Destination, Protocol, Length, Info

Labeling scheme:
  DDoS-TCP  : mixed capture (normal + SYN flood). Labeled by IP + time-window
              rule (DDOS_SRC / DDOS_DST / DDOS_T_START / DDOS_T_END).
  PortScan  : capture is pure port-scan traffic -> every row labeled 2.
  Fuzzer    : capture is pure fuzzing/malformed traffic -> every row labeled 3.
  BruteForce: capture is pure brute-force traffic -> every row labeled 4.
"""

from pathlib import Path

# ── Random seed ────────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Input CSV paths — adjust to your own capture locations ─────────────────
DDOS_TCP_CSV = "all_traffic.csv"           # contains normal + SYN-flood
PORTSCAN_CSV = "Port_Scanning.csv"
FUZZER_CSV = "Fuzzer_Attack.csv"
BRUTEFORCE_CSV = "Brute Force ATTACK.csv"

# ── Output paths ─────────────────────────────────────────────────────────
OUTPUT_DIR = Path("loads")
OUTPUT_FEATURES = OUTPUT_DIR / "features_final.csv"
OUTPUT_BUNDLE = OUTPUT_DIR / "ids_bundle.pkl"
EDA_RAW_PNG = "eda_raw.png"
EDA_FEATURES_PNG = "eda_features.png"
MODEL_COMPARISON_PNG = "model_comparison.png"
CONFUSION_MATRICES_PNG = "confusion_matrices.png"
FEATURE_IMPORTANCE_PNG = "feature_importance.png"

# ── DDoS-TCP labeling rule (SYN-flood window inside the mixed capture) ─────
DDOS_SRC = "10.0.0.5"
DDOS_DST = "10.0.0.3"
DDOS_T_START = 196.0
DDOS_T_END = 235.0

# ── Row cap applied when loading the PortScan capture ──────────────────────
PORTSCAN_NROWS = 300_000

# ── Class map ────────────────────────────────────────────────────────────
CLASS_NAMES = {
    0: "Normal",
    1: "DDoS",
    2: "PortScan",
    3: "Fuzzer",
    4: "BruteForce",
}

CLASS_COLORS = {
    0: "#639922",  # green   — safe
    1: "#E24B4A",  # red     — DDoS-TCP
    2: "#378ADD",  # blue    — PortScan
    3: "#7F77DD",  # purple  — Fuzzer
    4: "#D4537E",  # pink    — BruteForce
}

CLASS_FILL = {
    0: "#EAF3DE",
    1: "#FCEBEB",
    2: "#E6F1FB",
    3: "#EEEDFE",
    4: "#FBEAF0",
}

# ── Protocol numeric map (mirrors Wireshark display names) ─────────────────
PROTOCOL_MAP = {
    "ICMP": 1,
    "TCP": 6,
    "UDP": 17,
    "ICMPv6": 58,
    "ARP": 2054,
    "OpenFlow": 6653,
    "DNS": 53,
    "MDNS": 5353,
    "HTTP": 80,
}

# ── Sliding-window aggregation parameters ───────────────────────────────────
WINDOW_SIZE = 100  # packets per window
STEP_SIZE = 10     # slide step (90% overlap)

# ── Downsampling (applied to the merged raw-packet dataset before windowing)─
DOWNSAMPLE_FRAC = 0.1
DOWNSAMPLE_MIN_KEEP = 500_000
DOWNSAMPLE_LABELS = [0, 1, 4]  # Normal, DDoS-TCP, PortScan

# ── Train / test split ──────────────────────────────────────────────────────
TRAIN_FRACTION = 0.70  # per-class temporal split: first 70% train, last 30% test

# ── Window-level feature columns (order matters — must match training) ─────
FEATURE_COLS = [
    "pkt_count", "byte_total", "byte_mean", "byte_std", "byte_min", "byte_max",
    "pkt_rate", "byte_rate", "flow_duration",
    "IAT_mean", "IAT_std", "IAT_min", "IAT_max", "IAT_total",
    "SYN_count", "ACK_count", "FIN_count", "RST_count", "PSH_count",
    "syn_ratio", "ack_ratio", "fin_ratio", "rst_ratio", "psh_ratio",
    "syn_no_ack", "n_unique_src", "n_unique_dst", "n_unique_flows",
    "n_unique_dports", "n_unique_sports", "protocol_entropy",
]
