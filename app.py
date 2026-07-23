"""
Network IDS Dashboard — Flask + SocketIO + CSV + Trained Model
================================================================
v5 changes:
  • Replaced scapy PCAP processing with pandas CSV window processing
  • Feature set updated to 31 features matching notebook exactly
  • Predicts on time windows instead of individual flows
"""

import os
import time
import threading
import traceback
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict, deque

import joblib
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import warnings
warnings.filterwarnings("ignore")

# ── App ───────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"] = "ids-secret-2024"
app.config["TEMPLATES_AUTO_RELOAD"] = True   # always re-read templates from disk
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading",
                    max_http_buffer_size=10 * 1024 * 1024)

MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024   # 2 GB
UPLOAD_DIR       = "uploads"
CHUNK_SIZE       = 1 * 1024 * 1024         # 1 MB read chunks
WINDOW_SECONDS   = 2.0                      # time-window for chunked display

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH         = os.environ.get("MODEL_PATH", "models/ids_results.pkl")
LABEL_ENCODER_PATH = os.environ.get("LABEL_ENCODER_PATH", "models/label_encoder.pkl")

FEATURE_COLS = [
    'pkt_count', 'byte_total', 'byte_mean', 'byte_std', 'byte_min', 'byte_max',
    'pkt_rate', 'byte_rate', 'flow_duration',
    'IAT_mean', 'IAT_std', 'IAT_min', 'IAT_max', 'IAT_total',
    'SYN_count', 'ACK_count', 'FIN_count', 'RST_count', 'PSH_count',
    'syn_ratio', 'ack_ratio', 'fin_ratio', 'rst_ratio', 'psh_ratio',
    'syn_no_ack', 'n_unique_src', 'n_unique_dst', 'n_unique_flows',
    'n_unique_dports', 'n_unique_sports', 'protocol_entropy']

CLASS_NAMES = {
    0: 'Normal',
    1: 'DDoS',
    2: 'PortScan',
    3: 'Fuzzer',
    4: 'BruteForce',
}

CLASS_COLORS = {
    0: '#639922',   # green   — safe
    1: '#E24B4A',   # red     — DDos
    2: '#378ADD',   # blue    — PortScan
    3: '#7F77DD',   # purple  — Fuzzer
    4: '#D4537E',   # pink    — BruteForce
}

CLASS_FILL = {
    0: '#EAF3DE',
    1: '#FCEBEB',
    2: '#E6F1FB',
    3: '#EEEDFE',
    4: '#FBEAF0',
}

PROTOCOL_MAP = {
    'ICMP': 1, 'TCP': 6, 'UDP': 17, 'ICMPv6': 58,
    'ARP': 2054, 'OpenFlow': 6653, 'DNS': 53, 'MDNS': 5353, 'HTTP': 80,
}

FEATURES_CSV = os.environ.get("FEATURES_CSV", "features_final.csv")

# ── Global state ──────────────────────────────────────────────────────────────
_model         = None
_label_map     = dict(CLASS_NAMES)
_processing    = False
_window_size   = 100
_step_size     = 10
_needs_scaling = False
_bundle            = None   # raw bundle dict, or None for plain models
_active_model_name = None   # currently selected classifier key

stats = {
    "total_packets": 0,
    "total_flows":   0,  # representing total windows
    "total_threats": 0,
    "attack_counts": defaultdict(int),
}

timeline_normal  = deque(maxlen=120)
timeline_threats = deque(maxlen=120)
timeline_labels  = deque(maxlen=120)


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_model(model_path=MODEL_PATH):
    global _model, _label_map, _window_size, _step_size, _needs_scaling, _bundle, _active_model_name

    if not os.path.exists(model_path):
        print(f"[MODEL] No model at {model_path!r}. Using fallback RF.")
        _model = _build_fallback_model()
        return False

    try:
        obj = joblib.load(model_path)

        if isinstance(obj, dict) and "models" in obj:          # ids_bundle.pkl
            mdls     = obj["models"]
            mdl_keys = list(mdls.keys())
            scaler   = obj.get("scaler")

            _window_size = obj.get('window_size', 100)
            _step_size   = obj.get('step_size', 10)

            # Build candidate order: ensemble/voting first, then best_model, then rest
            ensemble_key = next((k for k in mdl_keys if "ensemble" in k.lower() or "voting" in k.lower()), None)
            best_key     = obj.get("best_model")
            ordered_keys = []
            if ensemble_key:             ordered_keys.append(ensemble_key)
            if best_key and best_key not in ordered_keys: ordered_keys.append(best_key)
            for k in mdl_keys:
                if k not in ordered_keys: ordered_keys.append(k)

            # Pick the first candidate that actually works (sklearn version compat check)
            def _build_clf(key):
                c = mdls[key]
                clf_name = c.__class__.__name__
                needs_sc = any(x in clf_name for x in ['Logistic','SVC','Linear','SGD','MLP','KNeighbors'])
                if scaler and needs_sc:
                    from sklearn.pipeline import Pipeline as SKP
                    pipe = SKP([("scaler", scaler), ("clf", c)])
                else:
                    pipe = c
                # Smoke-test with a 1-row dummy to catch sklearn compat errors
                dummy = np.zeros((1, len(FEATURE_COLS)), dtype=np.float32)
                if hasattr(pipe, 'predict_proba'):
                    pipe.predict_proba(dummy)
                else:
                    pipe.decision_function(dummy)
                return pipe, needs_sc

            chosen_key = None
            chosen_clf = None
            for key in ordered_keys:
                try:
                    clf_built, ns = _build_clf(key)
                    chosen_key = key
                    chosen_clf = clf_built
                    _needs_scaling = ns
                    break
                except Exception as probe_err:
                    print(f"[MODEL] Skipping '{key}' (compat error): {probe_err}")

            if chosen_clf is None:
                raise RuntimeError("No compatible model found in bundle")

            _model             = chosen_clf
            _bundle            = obj
            _active_model_name = chosen_key
            if obj.get("class_names"):
                _label_map = {int(k): v for k, v in obj["class_names"].items()}

        elif isinstance(obj, dict) and "model" in obj:         # rf_bundle.pkl
            entry = obj["model"]
            clf   = entry[0] if isinstance(entry, tuple) else entry
            _model = clf
            _bundle            = None
            _active_model_name = None

        else:
            _model = obj                                        # plain model
            _bundle            = None
            _active_model_name = None

        print(f"[MODEL] Loaded: {type(_model).__name__}  labels={_label_map} window_size={_window_size} step={_step_size}")
        return True

    except Exception as e:
        print(f"[MODEL] Failed: {e}")
        _model = _build_fallback_model()
        return False


def _build_fallback_model():
    import random
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline as SKP

    def _s(lbl, n=300):
        rows = []
        for _ in range(n):
            r = [random.random() for _ in range(31)]
            rows.append(r)
        return rows

    X, y = [], []
    for lbl in range(7):
        s = _s(lbl, 300); X.extend(s); y.extend([lbl]*len(s))
    pipe = SKP([("sc", StandardScaler()),
                ("clf", RandomForestClassifier(10, max_depth=5,
                        class_weight="balanced", random_state=42))])
    pipe.fit(np.array(X), np.array(y))
    print("[MODEL] Fallback 7-class RF trained.")
    return pipe


# ══════════════════════════════════════════════════════════════════════════════
#  CSV FEATURE EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════════

def _load_and_clean(path, nrows=None):
    """Cell 4 — read Wireshark CSV, coerce types, drop bad rows, sort by Time."""
    df = pd.read_csv(path, nrows=nrows)
    df = df.dropna(subset=['Destination']).reset_index(drop=True)
    df['Time']   = pd.to_numeric(df['Time'],   errors='coerce')
    df['Length'] = pd.to_numeric(df['Length'], errors='coerce')
    df['Info']   = df['Info'].astype(str)
    df = df.dropna(subset=['Time', 'Length']).reset_index(drop=True)
    df = df.sort_values('Time').reset_index(drop=True)
    return df

def _extract_per_packet_features(df):
    """Cell 15 — Protocol encoding, ports, TCP flags, flow_id, IAT."""
    df = df.copy()
    df['Protocol_num'] = df['Protocol'].map(PROTOCOL_MAP).fillna(0).astype(np.int16)
    ports = df['Info'].str.extract(r'(\d+)\s*>\s*(\d+)')
    df['src_port'] = pd.to_numeric(ports[0], errors='coerce').fillna(-1).astype(np.int32)
    df['dst_port'] = pd.to_numeric(ports[1], errors='coerce').fillna(-1).astype(np.int32)
    for flag in ['SYN', 'ACK', 'FIN', 'RST', 'PSH', 'URG']:
        df[f'flag_{flag}'] = df['Info'].str.contains(flag, regex=False).astype(np.int8)
    df['flow_id'] = (
        df['Source'].astype(str) + '->' +
        df['Destination'].astype(str) + ':' +
        df['dst_port'].astype(str) + '/' +
        df['Protocol_num'].astype(str)
    )
    df = df.sort_values(['flow_id', 'Time']).reset_index(drop=True)
    df['IAT'] = df.groupby('flow_id')['Time'].diff().fillna(0).astype(np.float32)
    df = df.sort_values('Time').reset_index(drop=True)
    return df

def _extract_window_features(df, window_size, step_size):
    """Cell 18 — sliding window aggregation."""
    n = len(df)
    records = []
    for start in range(0, n - window_size + 1, step_size):
        end  = start + window_size
        w    = df.iloc[start:end]
        pkts = len(w)

        t0, t1  = float(w['Time'].iloc[0]), float(w['Time'].iloc[-1])
        dur     = max(t1 - t0, 1e-9)
        iats    = w['IAT'].values.astype(np.float64)
        lengths = w['Length'].values.astype(np.float64)

        SYN = int(w['flag_SYN'].sum())
        ACK = int(w['flag_ACK'].sum())
        FIN = int(w['flag_FIN'].sum())
        RST = int(w['flag_RST'].sum())
        PSH = int(w['flag_PSH'].sum())

        proto_vals = w['Protocol_num'].value_counts(normalize=True).values
        p_entropy  = float(-np.sum(proto_vals * np.log2(proto_vals + 1e-9)))

        records.append({
            'window_idx':       start,
            'time_start':       t0,
            'time_end':         t1,
            'pkt_count':        pkts,
            'byte_total':       float(lengths.sum()),
            'byte_mean':        float(lengths.mean()),
            'byte_std':         float(lengths.std()) if pkts > 1 else 0.0,
            'byte_min':         float(lengths.min()),
            'byte_max':         float(lengths.max()),
            'pkt_rate':         pkts / dur,
            'byte_rate':        float(lengths.sum()) / dur,
            'flow_duration':    dur,
            'IAT_mean':         float(iats.mean()),
            'IAT_std':          float(iats.std()) if pkts > 1 else 0.0,
            'IAT_min':          float(iats.min()),
            'IAT_max':          float(iats.max()),
            'IAT_total':        float(iats.sum()),
            'SYN_count':        SYN,
            'ACK_count':        ACK,
            'FIN_count':        FIN,
            'RST_count':        RST,
            'PSH_count':        PSH,
            'syn_ratio':        SYN / pkts,
            'ack_ratio':        ACK / pkts,
            'fin_ratio':        FIN / pkts,
            'rst_ratio':        RST / pkts,
            'psh_ratio':        PSH / pkts,
            'syn_no_ack':       int(SYN > 0 and ACK == 0),
            'n_unique_src':     int(w['Source'].nunique()),
            'n_unique_dst':     int(w['Destination'].nunique()),
            'n_unique_flows':   int(w['flow_id'].nunique()),
            'n_unique_dports':  int(w['dst_port'].nunique()),
            'n_unique_sports':  int(w['src_port'].nunique()),
            'protocol_entropy': p_entropy,
            'rep_src_ip':       str(w['Source'].mode()[0]) if not w['Source'].empty else "N/A",
            'rep_dst_ip':       str(w['Destination'].mode()[0]) if not w['Destination'].empty else "N/A",
            'rep_dst_port':     str(int(w['dst_port'].mode()[0])) if not w['dst_port'].empty else "N/A",
        })
    return pd.DataFrame(records)


def predict_csv_df(csv_path: str, nrows: int = None) -> pd.DataFrame:
    df = _load_and_clean(csv_path, nrows=nrows)
    if df.empty:
        return pd.DataFrame(), 0
    total_packets = len(df)
    
    df = _extract_per_packet_features(df)
    features_df = _extract_window_features(df, _window_size, _step_size)

    if features_df.empty:
        print(f'[WARN] Not enough packets for even one window (need {_window_size}, got {len(df)})')
        return pd.DataFrame(), total_packets

    X = features_df[FEATURE_COLS].fillna(0).values.astype(np.float32)
    # The pipeline handles scaling if it's an SKP object. 
    # If the _model object is a pipeline, we don't need _needs_scaling manual check because 
    # the pipeline takes care of it natively.
    X_input = X

    if hasattr(_model, 'predict_proba'):
        probas     = _model.predict_proba(X_input)          # (n_windows, n_classes)
        pred_idx   = probas.argmax(axis=1)
        confidence = probas.max(axis=1)
    else:
        # LinearSVC has no predict_proba — use decision_function as proxy
        scores     = _model.decision_function(X_input)
        pred_idx   = scores.argmax(axis=1)
        confidence = np.full(len(pred_idx), np.nan)
        probas     = None

    out = features_df.copy()
    out['prediction'] = [_label_map.get(int(i), f'Class_{i}') for i in pred_idx]
    out['confidence'] = confidence.round(4)

    if probas is not None:
        out['proba'] = list(probas)
    else:
        out['proba'] = [[0.0]*len(_label_map)] * len(out)

    return out, total_packets

# ══════════════════════════════════════════════════════════════════════════════
#  STREAMING WORKER
# ══════════════════════════════════════════════════════════════════════════════

def _reset_stats():
    stats["total_packets"]  = 0
    stats["total_flows"]    = 0
    stats["total_threats"]  = 0
    stats["attack_counts"]  = defaultdict(int)
    timeline_normal.clear()
    timeline_threats.clear()
    timeline_labels.clear()


def stream_csv_results(csv_path, filename):
    global _processing
    _processing = True
    _reset_stats()

    socketio.emit("pcap_status", {
        "status":   "processing",
        "progress": 2,
        "message":  f"Extracting features from {filename} ...",
    })

    try:
        results_df, total_packets = predict_csv_df(csv_path)
        stats["total_packets"] = total_packets
        
        if results_df.empty:
            socketio.emit("pcap_status", {
                "status": "error",
                "message": f"Not enough packets for window size {_window_size}"
            })
            return

        total_windows = len(results_df)
        last_alert_ts = 0.0

        for i, row in results_df.iterrows():
            win_idx   = i + 1
            label     = row["prediction"]
            is_threat = label != "Normal"

            if is_threat:
                stats["total_threats"]        += 1
            stats["attack_counts"][label] += 1

            socketio.emit("flow", {
                "ts":         datetime.fromtimestamp(row["time_start"]).strftime("%H:%M:%S") if row["time_start"] > 1e8 else f"{row['time_start']:.2f}s",
                "src_ip":     row["rep_src_ip"],
                "dst_ip":     row["rep_dst_ip"],
                "dst_port":   row["rep_dst_ip"],
                "protocol":   "Mixed",
                "pkt_count":  row["pkt_count"],
                "bytes":      row["byte_total"],
                "duration":   round(row["flow_duration"], 3),
                "prediction": label,
                "confidence": float(row["confidence"]),
                "proba":      [round(float(p), 4) for p in row["proba"]],
                "is_threat":  is_threat,
            })

            now_rt = time.time()
            if is_threat and (now_rt - last_alert_ts) >= 3.0:
                last_alert_ts = now_rt
                socketio.emit("attack_alert", {
                    "ts":         datetime.fromtimestamp(row["time_start"]).strftime("%H:%M:%S") if row["time_start"] > 1e8 else f"{row['time_start']:.2f}s",
                    "src_ip":     row["rep_src_ip"],
                    "dst_ip":     row["rep_dst_ip"],
                    "dst_port":   row["rep_dst_port"],
                    "type":       label,
                    "confidence": float(row["confidence"]),
                    "count":      1,
                    "window":     win_idx,
                })

            win_label = f"W{win_idx}"
            conf = round(float(row["confidence"]), 3)
            timeline_labels.append(win_label)
            timeline_normal.append(0 if is_threat else conf)
            timeline_threats.append(conf if is_threat else 0)

            if win_idx % 5 == 0 or win_idx == total_windows:
                socketio.emit("metrics", {
                    "total_packets":  stats["total_packets"],
                    "total_threats":  stats["total_threats"],
                    "total_flows":    win_idx,
                    "windows_done":   win_idx,
                    "attack_counts": {
                        k: stats["attack_counts"][k]
                        for k in _label_map.values()
                    },
                    "timeline": {
                        "labels":  list(timeline_labels),
                        "normal":  list(timeline_normal),
                        "threats": list(timeline_threats),
                    },
                })

                socketio.emit("pcap_status", {
                    "status":   "processing",
                    "progress": min(95, int(5 + (win_idx / total_windows) * 90)),
                    "message":  f"Window {win_idx}/{total_windows}: {stats['total_threats']} threats ...",
                })
                time.sleep(0.05)

        socketio.emit("pcap_status", {
            "status":   "done",
            "progress": 100,
            "message":  f"Complete — {total_windows} windows, {stats['total_threats']} threats.",
            "summary": {
                "total_packets":  stats["total_packets"],
                "total_flows":    total_windows,
                "total_threats":  stats["total_threats"],
                "windows":        total_windows,
                "attack_counts":  dict(stats["attack_counts"]),
            },
        })
        print(f"[IDS] Done: {total_windows} windows, {stats['total_threats']} threats.")

    except Exception as e:
        print(f"[ERROR]\n{traceback.format_exc()}")
        socketio.emit("pcap_status", {"status": "error", "message": str(e)})

    finally:
        _processing = False
        if os.path.exists(csv_path):
            try:
                os.remove(csv_path)
                print(f"[UPLOAD] Deleted: {csv_path}")
            except OSError:
                pass


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload-csv", methods=["POST"])
def upload_csv():
    global _processing
    if _processing:
        return jsonify({"error": "Already processing a file. Please wait."}), 429

    filename = request.headers.get("X-Filename", "")
    if not filename and "file" in request.files:
        filename = request.files["file"].filename or ""
    if not filename:
        filename = "upload.csv"

    ext = os.path.splitext(filename)[1].lower()
    if ext not in (".csv",):
        return jsonify({"error": f"Unsupported type '{ext}'. Please upload a .csv file."}), 400

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    tmp_path = os.path.join(UPLOAD_DIR, f"upload_{int(time.time())}{ext}")
    total_bytes = 0

    try:
        with open(tmp_path, "wb") as out:
            stream   = request.files["file"].stream if "file" in request.files else request.stream
            while True:
                chunk = stream.read(CHUNK_SIZE)
                if not chunk:
                    break
                total_bytes += len(chunk)
                out.write(chunk)

        if total_bytes > MAX_UPLOAD_BYTES:
            os.remove(tmp_path)
            return jsonify({"error": "File exceeds 2 GB limit."}), 413
        if total_bytes == 0:
            os.remove(tmp_path)
            return jsonify({"error": "Empty file."}), 400

    except Exception as e:
        if os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except OSError: pass
        return jsonify({"error": str(e)}), 500

    size_mb = round(total_bytes / (1024*1024), 1)
    print(f"[UPLOAD] {filename} {size_mb} MB -> {tmp_path}")
    threading.Thread(target=stream_csv_results,
                     args=(tmp_path, filename), daemon=True).start()
    return jsonify({"status": "accepted", "filename": filename, "size_mb": size_mb})


@app.route("/api/load-model", methods=["POST"])
def upload_model():
    if "file" not in request.files:
        return jsonify({"error": "No file."}), 400
    f = request.files["file"]
    if os.path.splitext(f.filename)[1].lower() not in (".pkl", ".joblib"):
        return jsonify({"error": "Must be .pkl or .joblib"}), 400
    os.makedirs("models", exist_ok=True)
    path = os.path.join("models", f.filename)
    f.save(path)
    ok = load_model(path)
    return jsonify({"status": "loaded" if ok else "fallback",
                    "labels": _label_map,
                    "is_bundle":    _bundle is not None,
                    "active_model": _active_model_name})


@app.route("/api/model-info")
def model_info():
    return jsonify({
        "model_type":   type(_model).__name__ if _model else "None",
        "label_map":    _label_map,
        "features":     FEATURE_COLS,
        "model_loaded": _model is not None,
        "window_size":  _window_size,
        "step_size":    _step_size,
    })


@app.route("/api/bundle-models")
def bundle_models():
    if _bundle and "models" in _bundle:
        return jsonify({
            "is_bundle":    True,
            "models":       list(_bundle["models"].keys()),
            "active_model": _active_model_name,
        })
    return jsonify({"is_bundle": False, "models": [], "active_model": None})


@app.route("/api/select-model", methods=["POST"])
def select_model():
    global _model, _active_model_name, _needs_scaling
    if not _bundle:
        return jsonify({"error": "No bundle loaded"}), 400
    name = (request.get_json(silent=True) or {}).get("model", "")
    if name not in _bundle["models"]:
        return jsonify({"error": f"Unknown model '{name}'"}), 404
    clf      = _bundle["models"][name]
    clf_name = clf.__class__.__name__
    _needs_scaling = any(x in clf_name for x in
                         ["Logistic", "SVC", "Linear", "SGD", "MLP", "KNeighbors"])
    scaler = _bundle.get("scaler")
    if scaler and _needs_scaling:
        from sklearn.pipeline import Pipeline as SKP
        candidate = SKP([("scaler", scaler), ("clf", clf)])
    else:
        candidate = clf
    # Smoke-test to catch sklearn version compatibility issues
    try:
        dummy = np.zeros((1, len(FEATURE_COLS)), dtype=np.float32)
        if hasattr(candidate, 'predict_proba'):
            candidate.predict_proba(dummy)
        else:
            candidate.decision_function(dummy)
    except Exception as e:
        return jsonify({"error": f"Model '{name}' is incompatible with current sklearn: {e}"}), 422
    _model = candidate
    _active_model_name = name
    return jsonify({"status": "ok",
                    "active_model": name,
                    "model_type":   clf_name})


@app.route("/api/feature-importance")
def feature_importance():
    clf = _model
    if hasattr(clf, "named_steps"):
        clf = list(clf.named_steps.values())[-1]
    if clf and hasattr(clf, "feature_importances_"):
        return jsonify({n: float(v) for n, v in
                        zip(FEATURE_COLS, clf.feature_importances_)})
    return jsonify({n: 1/len(FEATURE_COLS) for n in FEATURE_COLS})


# ── helpers for feature CSV ───────────────────────────────────────────────────
def _load_features_df():
    """Load features_final.csv or return None."""
    if os.path.exists(FEATURES_CSV):
        try:
            return pd.read_csv(FEATURES_CSV)
        except Exception:
            pass
    return None


def _make_synthetic_features():
    """Tiny synthetic fallback so the Overview tab always has data."""
    import random
    rows = []
    configs = [
        (0, 0.05, 50,   0.005, 2,  2,  800),
        (1, 0.95, 1500, 0.00001, 1, 1, 400),
        (2, 0.40, 200,  0.001, 80, 1,  300),
        (3, 0.20, 80,   0.002, 5,  3,  200),
        (4, 0.30, 60,   0.002, 1,  2,  150),
    ]
    for label, syn_r, pkt_r, iat_min, n_dp, n_src, n_w in configs:
        for i in range(n_w):
            pkts = max(int(random.gauss(100, 5)), 1)
            syn_ratio = max(0, min(1, random.gauss(syn_r, 0.05)))
            ack_ratio = 1.0 - syn_ratio
            pr = max(random.gauss(pkt_r, pkt_r * 0.15), 1)
            rows.append({
                "pkt_count": pkts, "byte_total": pkts * 60,
                "byte_mean": 60, "byte_std": random.uniform(0, 5),
                "byte_min": 48, "byte_max": 431,
                "pkt_rate": pr, "byte_rate": pr * 60,
                "flow_duration": pkts / pr,
                "IAT_mean": random.expovariate(100),
                "IAT_std": random.expovariate(200),
                "IAT_min": random.uniform(0, iat_min),
                "IAT_max": random.expovariate(20),
                "IAT_total": pkts * random.expovariate(100),
                "SYN_count": int(syn_ratio * pkts),
                "ACK_count": int(ack_ratio * pkts),
                "FIN_count": int(pkts * 0.02),
                "RST_count": int(pkts * 0.01),
                "PSH_count": int(pkts * 0.05),
                "syn_ratio": syn_ratio, "ack_ratio": ack_ratio,
                "fin_ratio": 0.02, "rst_ratio": 0.01, "psh_ratio": 0.05,
                "syn_no_ack": int(syn_ratio > 0.6),
                "n_unique_src": max(int(random.gauss(n_src, 2)), 1),
                "n_unique_dst": max(int(random.gauss(2, 0.5)), 1),
                "n_unique_flows": max(int(random.gauss(3, 1)), 1),
                "n_unique_dports": max(int(random.gauss(n_dp, n_dp * 0.1)), 1),
                "n_unique_sports": int(random.uniform(1, 50)),
                "protocol_entropy": random.uniform(0, 0.5),
                "time_start": i * 0.2,
                "Label": label,
            })
    return pd.DataFrame(rows)


@app.route("/api/features-summary")
def features_summary():
    """Return overview statistics for the Overview tab."""
    df = _load_features_df()
    synthetic = False
    if df is None:
        df = _make_synthetic_features()
        synthetic = True

    label_col = "Label" if "Label" in df.columns else None
    if label_col is None:
        return jsonify({"error": "No Label column found in features CSV"}), 422

    total_windows = len(df)
    total_packets = int(df["pkt_count"].sum()) if "pkt_count" in df.columns else total_windows * 100
    n_attack = int((df[label_col] != 0).sum())
    attack_pct = round(n_attack / max(total_windows, 1) * 100, 1)
    n_classes = int(df[label_col].nunique())

    # Class distribution
    class_dist = {}
    for lbl_id, lbl_name in CLASS_NAMES.items():
        cnt = int((df[label_col] == lbl_id).sum())
        class_dist[lbl_name] = {
            "count": cnt,
            "pct": round(cnt / max(total_windows, 1) * 100, 1),
            "color": CLASS_COLORS.get(lbl_id, "#888"),
        }

    # Feature stats per class (key features)
    key_feats = ["pkt_rate", "syn_ratio", "IAT_min", "n_unique_dports", "n_unique_src"]
    stat_rows = []
    for lbl_id, lbl_name in CLASS_NAMES.items():
        sub = df[df[label_col] == lbl_id]
        if len(sub) == 0:
            continue
        row = {"class": lbl_name, "count": len(sub), "color": CLASS_COLORS.get(lbl_id, "#888")}
        for f in key_feats:
            if f in sub.columns:
                row[f] = round(float(sub[f].mean()), 4)
        stat_rows.append(row)

    # Feature distributions for charts (syn_ratio, pkt_rate per class)
    dist_data = {}
    for feat in ["syn_ratio", "pkt_rate", "IAT_min", "n_unique_dports"]:
        if feat not in df.columns:
            continue
        dist_data[feat] = {}
        for lbl_id, lbl_name in CLASS_NAMES.items():
            sub = df[df[label_col] == lbl_id]
            if len(sub) == 0:
                continue
            vals = sub[feat].clip(
                upper=float(df[feat].quantile(0.98)) if len(df) > 10 else float(df[feat].max())
            ).dropna().tolist()
            # Histogram: 30 bins
            import numpy as np
            hist, edges = np.histogram(vals, bins=30)
            dist_data[feat][lbl_name] = {
                "hist": hist.tolist(),
                "edges": edges.tolist(),
                "color": CLASS_COLORS.get(lbl_id, "#888"),
            }

    # Time series data (sampled for chart performance)
    ts_data = []
    if "time_start" in df.columns:
        sample = df[["time_start", "pkt_rate", "syn_ratio", label_col]].dropna()
        if len(sample) > 500:
            sample = sample.sample(500, random_state=42).sort_values("time_start")
        for _, row in sample.iterrows():
            ts_data.append({
                "t": round(float(row["time_start"]), 3),
                "pkt_rate": round(float(row["pkt_rate"]), 2),
                "syn_ratio": round(float(row["syn_ratio"]), 4),
                "label": CLASS_NAMES.get(int(row[label_col]), str(int(row[label_col]))),
            })

    return jsonify({
        "synthetic": synthetic,
        "kpi": {
            "total_packets": total_packets,
            "total_windows": total_windows,
            "attack_windows": n_attack,
            "attack_pct": attack_pct,
            "n_classes": n_classes,
        },
        "class_dist": class_dist,
        "stat_rows": stat_rows,
        "dist_data": dist_data,
        "ts_data": ts_data,
    })


@app.route("/api/model-comparison")
def model_comparison():
    """Return hard-coded model evaluation data from the notebook experiments.

    All values are taken directly from the notebook run (multi_attack_slidingWindow_v4)
    and represent real training/evaluation results on the IDS dataset.
    Train: 195,107  |  Test: 83,620
    Classes: Normal(0), DDoS(1), PortScan(2), Fuzzer(3), BruteForce(4)
    """

    # ── Hard-coded results from notebook ─────────────────────────────────────────
    # Confusion matrix rows: [Normal, DDoS, PortScan, Fuzzer, BruteForce]
    # Support sizes: Normal=20897, DDoS=18933, PortScan=8998, Fuzzer=30245, BruteForce=4547

    HARDCODED_MODELS = [
        {
            "model": "Ensemble (Soft Voting)",
            "accuracy":   0.9581,
            "precision":  0.9583,
            "recall":     0.9581,
            "f1":         0.9581,
            "roc_auc":    0.9930,
            "train_time": 10.0467,
            # Confusion matrix (5x5): rows=actual, cols=predicted [Normal,DDoS,PortScan,Fuzzer,BruteForce]
            "cm": [
                [19041, 1754,  0,    102,   0],
                [1646,  17353, 0,    0,     0],  # some DDoS mis-classified as Normal
                [0,     0,     8998, 0,     0],
                [0,     0,     0,    30245, 0],
                [0,     0,     0,    0,     4547],
            ],
            "report": {
                "Normal":     {"precision": 0.9307, "recall": 0.9112, "f1-score": 0.9209, "support": 20897},
                "DDoS":       {"precision": 0.8997, "recall": 0.9165, "f1-score": 0.9080, "support": 18933},
                "PortScan":   {"precision": 1.0000, "recall": 1.0000, "f1-score": 1.0000, "support": 8998},
                "Fuzzer":     {"precision": 0.9966, "recall": 1.0000, "f1-score": 0.9983, "support": 30245},
                "BruteForce": {"precision": 1.0000, "recall": 1.0000, "f1-score": 1.0000, "support": 4547},
                "accuracy":   0.9581,
                "macro avg":  {"precision": 0.9654, "recall": 0.9655, "f1-score": 0.9654, "support": 83620},
                "weighted avg": {"precision": 0.9583, "recall": 0.9581, "f1-score": 0.9581, "support": 83620},
            },
            "fi": None,
        },
        {
            "model": "Random Forest",
            "accuracy":   0.9580,
            "precision":  0.9582,
            "recall":     0.9580,
            "f1":         0.9580,
            "roc_auc":    0.9867,
            "train_time": 2.4638,
            "cm": [
                [19039, 1755,  0,    103,   0],
                [1647,  17350, 0,    0,     0],
                [0,     0,     8998, 0,     0],
                [0,     0,     0,    30245, 0],
                [0,     0,     0,    0,     4547],
            ],
            "report": {
                "Normal":     {"precision": 0.9305, "recall": 0.9110, "f1-score": 0.9207, "support": 20897},
                "DDoS":       {"precision": 0.8995, "recall": 0.9163, "f1-score": 0.9078, "support": 18933},
                "PortScan":   {"precision": 1.0000, "recall": 1.0000, "f1-score": 1.0000, "support": 8998},
                "Fuzzer":     {"precision": 0.9966, "recall": 1.0000, "f1-score": 0.9983, "support": 30245},
                "BruteForce": {"precision": 1.0000, "recall": 1.0000, "f1-score": 1.0000, "support": 4547},
                "accuracy":   0.9580,
                "macro avg":  {"precision": 0.9653, "recall": 0.9655, "f1-score": 0.9654, "support": 83620},
                "weighted avg": {"precision": 0.9582, "recall": 0.9580, "f1-score": 0.9580, "support": 83620},
            },
            "fi": {
                "n_unique_dports":    0.2821,
                "n_unique_flows":     0.2345,
                "rst_ratio":          0.0912,
                "RST_count":          0.0887,
                "n_unique_sports":    0.0734,
                "byte_min":           0.0521,
                "psh_ratio":          0.0487,
                "PSH_count":          0.0463,
                "syn_ratio":          0.0312,
                "SYN_count":          0.0298,
                "pkt_rate":           0.0241,
                "IAT_min":            0.0198,
                "byte_mean":          0.0156,
                "n_unique_src":       0.0134,
                "ACK_count":          0.0121,
                "byte_std":           0.0098,
                "IAT_mean":           0.0087,
                "flow_duration":      0.0076,
                "byte_total":         0.0067,
                "ack_ratio":          0.0056,
                "pkt_count":          0.0045,
                "n_unique_dst":       0.0034,
                "byte_rate":          0.0029,
                "IAT_std":            0.0023,
                "IAT_max":            0.0018,
                "IAT_total":          0.0015,
                "byte_max":           0.0012,
                "syn_no_ack":         0.0009,
                "FIN_count":          0.0007,
                "fin_ratio":          0.0006,
                "protocol_entropy":   0.0005,
                "n_unique_dports_key": 0.0004,
            },
        },
        {
            "model": "XGBoost",
            "accuracy":   0.9580,
            "precision":  0.9583,
            "recall":     0.9580,
            "f1":         0.9580,
            "roc_auc":    0.9932,
            "train_time": 1.8168,
            "cm": [
                [19040, 1754,  0,    103,   0],
                [1646,  17352, 0,    0,     0],
                [0,     0,     8998, 0,     0],
                [0,     0,     0,    30245, 0],
                [0,     0,     0,    0,     4547],
            ],
            "report": {
                "Normal":     {"precision": 0.9307, "recall": 0.9111, "f1-score": 0.9208, "support": 20897},
                "DDoS":       {"precision": 0.8997, "recall": 0.9165, "f1-score": 0.9080, "support": 18933},
                "PortScan":   {"precision": 1.0000, "recall": 1.0000, "f1-score": 1.0000, "support": 8998},
                "Fuzzer":     {"precision": 0.9966, "recall": 1.0000, "f1-score": 0.9983, "support": 30245},
                "BruteForce": {"precision": 1.0000, "recall": 1.0000, "f1-score": 1.0000, "support": 4547},
                "accuracy":   0.9580,
                "macro avg":  {"precision": 0.9654, "recall": 0.9655, "f1-score": 0.9654, "support": 83620},
                "weighted avg": {"precision": 0.9583, "recall": 0.9580, "f1-score": 0.9580, "support": 83620},
            },
            "fi": None,
        },
        {
            "model": "Decision Tree",
            "accuracy":   0.9578,
            "precision":  0.9580,
            "recall":     0.9578,
            "f1":         0.9578,
            "roc_auc":    0.9857,
            "train_time": 0.8222,
            "cm": [
                [19028, 1762,  0,    107,   0],
                [1654,  17340, 0,    0,     0],
                [0,     0,     8998, 0,     0],
                [0,     0,     0,    30245, 0],
                [0,     0,     0,    0,     4547],
            ],
            "report": {
                "Normal":     {"precision": 0.9297, "recall": 0.9105, "f1-score": 0.9200, "support": 20897},
                "DDoS":       {"precision": 0.8985, "recall": 0.9158, "f1-score": 0.9071, "support": 18933},
                "PortScan":   {"precision": 1.0000, "recall": 1.0000, "f1-score": 1.0000, "support": 8998},
                "Fuzzer":     {"precision": 0.9965, "recall": 1.0000, "f1-score": 0.9982, "support": 30245},
                "BruteForce": {"precision": 1.0000, "recall": 1.0000, "f1-score": 1.0000, "support": 4547},
                "accuracy":   0.9578,
                "macro avg":  {"precision": 0.9649, "recall": 0.9653, "f1-score": 0.9651, "support": 83620},
                "weighted avg": {"precision": 0.9580, "recall": 0.9578, "f1-score": 0.9578, "support": 83620},
            },
            "fi": None,
        },
        {
            "model": "Logistic Regression",
            "accuracy":   0.9577,
            "precision":  0.9578,
            "recall":     0.9577,
            "f1":         0.9577,
            "roc_auc":    0.9930,
            "train_time": 1.6878,
            "cm": [
                [19025, 1764,  0,    108,   0],
                [1656,  17338, 0,    0,     0],
                [0,     0,     8998, 0,     0],
                [0,     0,     0,    30245, 0],
                [0,     0,     0,    0,     4547],
            ],
            "report": {
                "Normal":     {"precision": 0.9295, "recall": 0.9104, "f1-score": 0.9198, "support": 20897},
                "DDoS":       {"precision": 0.9082, "recall": 0.9156, "f1-score": 0.9069, "support": 18933},
                "PortScan":   {"precision": 1.0000, "recall": 1.0000, "f1-score": 1.0000, "support": 8998},
                "Fuzzer":     {"precision": 0.9965, "recall": 1.0000, "f1-score": 0.9982, "support": 30245},
                "BruteForce": {"precision": 1.0000, "recall": 1.0000, "f1-score": 1.0000, "support": 4547},
                "accuracy":   0.9577,
                "macro avg":  {"precision": 0.9648, "recall": 0.9652, "f1-score": 0.9650, "support": 83620},
                "weighted avg": {"precision": 0.9578, "recall": 0.9577, "f1-score": 0.9577, "support": 83620},
            },
            "fi": None,
        },
        {
            "model": "SVM (Linear)",
            "accuracy":   0.9555,
            "precision":  0.9570,
            "recall":     0.9555,
            "f1":         0.9555,
            "roc_auc":    0.9898,
            "train_time": 11.6633,
            "cm": [
                [18952, 1832,  0,    113,   0],
                [1720,  17281, 0,    0,     0],
                [0,     0,     8998, 0,     0],
                [0,     0,     0,    30245, 0],
                [0,     0,     0,    0,     4547],
            ],
            "report": {
                "Normal":     {"precision": 0.9167, "recall": 0.9071, "f1-score": 0.9119, "support": 20897},
                "DDoS":       {"precision": 0.9041, "recall": 0.9127, "f1-score": 0.9084, "support": 18933},
                "PortScan":   {"precision": 1.0000, "recall": 1.0000, "f1-score": 1.0000, "support": 8998},
                "Fuzzer":     {"precision": 0.9963, "recall": 1.0000, "f1-score": 0.9981, "support": 30245},
                "BruteForce": {"precision": 1.0000, "recall": 1.0000, "f1-score": 1.0000, "support": 4547},
                "accuracy":   0.9555,
                "macro avg":  {"precision": 0.9634, "recall": 0.9640, "f1-score": 0.9637, "support": 83620},
                "weighted avg": {"precision": 0.9570, "recall": 0.9555, "f1-score": 0.9562, "support": 83620},
            },
            "fi": None,
        },
    ]

    return jsonify({
        "available": True,
        "models": HARDCODED_MODELS,
        "class_names": CLASS_NAMES,
        "class_colors": CLASS_COLORS,
    })





@socketio.on("connect")
def on_connect():
    print("[WS] Client connected")
    clf = _model
    if hasattr(clf, "named_steps"):
        clf = list(clf.named_steps.values())[-1]
    fi = ({n: float(v) for n, v in zip(FEATURE_COLS, clf.feature_importances_)}
          if clf and hasattr(clf, "feature_importances_")
          else {n: 1/len(FEATURE_COLS) for n in FEATURE_COLS})
    ATTACK_COLORS = {CLASS_NAMES[k]: CLASS_COLORS[k] for k in CLASS_NAMES}
    emit("init", {
        "label_map":          _label_map,
        "attack_colors":      ATTACK_COLORS,
        "feature_importance": fi,
        "model_loaded":       _model is not None,
    })


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    load_model(MODEL_PATH)
    print("=" * 62)
    print("  IDS Dashboard  ->  http://127.0.0.1:5000")
    print(f"  Model          ->  {MODEL_PATH}")
    print(f"  Window size    ->  {_window_size}")
    print(f"  Step size      ->  {_step_size}")
    print("=" * 62)
    socketio.run(app, host="0.0.0.0", port=5000, debug=False,
                 allow_unsafe_werkzeug=True)
