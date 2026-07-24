# Machine Learning-Based Intrusion Detection

A graduation project that develops and evaluates a **five-class machine-learning intrusion detection workflow** using traffic generated in an emulated Software-Defined Networking environment.

The repository covers SDN traffic generation, packet capture, data cleaning and labeling, packet-based sliding-window feature extraction, machine-learning training and evaluation, model serialization, and a Flask dashboard for analysing previously recorded packet-level CSV files.

> **Project status:** Academic graduation project and reproducibility archive  
> **Validated deployment scope:** Offline CSV-based inference and visualisation  
> **Not validated as part of the final workflow:** Direct live controller-side inference and automatic OpenFlow mitigation

---

## Table of Contents

- [Project Overview](#project-overview)
- [Traffic Classes](#traffic-classes)
- [System Workflow](#system-workflow)
- [Repository Structure](#repository-structure)
- [Dataset and Feature Extraction](#dataset-and-feature-extraction)
- [Machine-Learning Models](#machine-learning-models)
- [Dashboard](#dashboard)
- [Installation](#installation)
- [Running the Dashboard](#running-the-dashboard)
- [Running the SDN Topology](#running-the-sdn-topology)
- [Controller Files](#controller-files)
- [Known Limitations](#known-limitations)
- [Security Notice](#security-notice)
- [Project Team](#project-team)
- [Repository History](#repository-history)
- [License](#license)

---

## Project Overview

The project investigates machine-learning-based intrusion detection for network traffic generated in an emulated SDN environment.

The implemented workflow includes:

1. Creating an SDN topology using Mininet and Open vSwitch.
2. Connecting the topology to a remote Ryu controller.
3. Generating normal and malicious traffic.
4. Capturing packet-level traffic and converting it to CSV.
5. Cleaning and labeling the collected traffic.
6. Building a global chronological timeline.
7. Extracting overlapping packet-window features.
8. Training and comparing several machine-learning classifiers.
9. Exporting trained models in a reusable model bundle.
10. Analysing packet-level CSV files through a Flask and Socket.IO dashboard.

The final dashboard processes previously recorded packet-level CSV files. It is not a direct live feed from the Ryu controller.

---

## Traffic Classes

| Label | Class |
|---:|---|
| 0 | Normal |
| 1 | DDoS |
| 2 | PortScan |
| 3 | Fuzzer |
| 4 | BruteForce |

---

## System Workflow

```text
Mininet / Open vSwitch / Ryu
              |
              v
      Traffic generation
              |
              v
 Packet capture and CSV export
              |
              v
 Cleaning, labeling, and merging
              |
              v
 100-packet sliding windows
      with a 10-packet step
              |
              v
   31 model input features
              |
              v
 Model training and evaluation
              |
              v
       ids_bundle.pkl
              |
              v
 Flask dashboard CSV inference
```

---

## Repository Structure

```text
machine-learning-based-ids/
|
├── README.md
├── AUTHORS.md
├── HISTORY.md
├── requirements.txt
├── requirements-lock.txt
├── .gitignore
├── .gitattributes
|
├── app.py
├── features_final.csv
|
├── templates/
│   └── index.html
|
├── models/
│   └── ids_bundle.pkl
|
├── notebooks/
│   ├── README.md
│   └── multi_attack_slidingWindow_v4.ipynb
|
├── modules/
|   ├── config.py            # paths, labeling rules, class maps, feature list — edit this first
|   ├── data_loading.py      # load/clean Wireshark CSVs, label, merge, downsample
|   ├── features.py          # per-packet feature extraction + sliding-window aggregation
|   ├── utils.py             # shared threshold/bias decision-rule helpers
|   ├── eda.py               # optional plotting (raw traffic + feature EDA, leakage check)
|   ├── train.py             # model zoo, evaluation, ensemble, bundle export — full pipeline entrypoint
|   ├── predict.py           # run a saved bundle against a new capture — CLI entrypoint
|   └── requirements.txt
|
|
├── sdn/
│   ├── README.md
│   └── topology/
│       └── topo.py
|
├── legacy/
│   └── flow-statistics-collector/
│       ├── README.md
│       └── Controller_Benign_Traffic.py
|
├── experimental/
│   └── online-ids-controller/
│       ├── README.md
│       └── traffic_controller.py
|
├── docs/
│   └── thesis/
│       ├── main.pdf
│       └── source-incomplete/
│           ├── README.md
│           ├── main.tex
│           └── chapter_*.tex
|
└── uploads/
    └── generated locally and excluded from Git
```

Large files such as `features_final.csv` and `models/ids_bundle.pkl` should be stored using **Git Large File Storage (Git LFS)**.

---

## Dataset and Feature Extraction

### Raw packet-level input

The dashboard expects a Wireshark or TShark-style CSV containing:

```text
Time
Source
Destination
Protocol
Length
Info
```

### Sliding-window configuration

- Window size: **100 packets**
- Step size: **10 packets**
- Window overlap: **90%**
- Classification type: **single-label, five-class**
- Final model inputs: **31 features**

### Model input features

```text
pkt_count
byte_total
byte_mean
byte_std
byte_min
byte_max
pkt_rate
byte_rate
flow_duration
IAT_mean
IAT_std
IAT_min
IAT_max
IAT_total
SYN_count
ACK_count
FIN_count
RST_count
PSH_count
syn_ratio
ack_ratio
fin_ratio
rst_ratio
psh_ratio
syn_no_ack
n_unique_src
n_unique_dst
n_unique_flows
n_unique_dports
n_unique_sports
protocol_entropy
```

The processed window-level dataset is stored as:

```text
features_final.csv
```

This file should **not** be uploaded to the Flow Inspector as a raw packet capture because its features have already been extracted.

---

## Machine-Learning Models

The experimental workflow includes:

- Decision Tree
- Random Forest
- Logistic Regression
- Linear Support Vector Machine
- XGBoost
- Soft-voting ensemble

The model bundle is stored as:

```text
models/ids_bundle.pkl
```

The bundle may contain trained models, scaling information, class mappings, window settings, and evaluation metadata. The dashboard attempts to select an ensemble or voting model first when one is available, while also allowing other compatible models to be selected.

### Evaluation outputs

The project reports:

- Accuracy
- Weighted precision
- Weighted recall
- Weighted F1-score
- Macro ROC-AUC
- Confusion matrices
- Training time
- Prediction time
- Feature importance

The notebook is the primary record of data preparation, model training, and evaluation experiments.

---

## Dashboard

The dashboard is implemented using:

- Flask
- Flask-SocketIO
- pandas
- NumPy
- scikit-learn
- joblib
- Chart.js
- Socket.IO

It provides:

- CSV upload and analysis
- Window-by-window predictions
- Threat alerts
- Class-distribution charts
- Dataset overview
- Feature visualisations
- Feature importance
- Model comparison
- Confusion matrices
- Classification reports

Backend:

```text
app.py
```

Frontend:

```text
templates/index.html
```

---

## Installation

### Requirements

- Python 3
- Git
- Git LFS
- A scikit-learn version compatible with the serialized model bundle

For the SDN topology:

### Clone the repository

```bash
git clone https://github.com/alyaaahmed019/machine-learning-based-ids.git
cd machine-learning-based-ids
```

Initialize Git LFS:

```bash
git lfs install
git lfs pull
```

### Create a virtual environment

#### Windows using Git Bash

```bash
py -m venv .venv
source .venv/Scripts/activate
```

#### Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

When an exact tested environment is available:

```bash
pip install -r requirements-lock.txt
```

---

## Running the Dashboard

Confirm that these files exist:

```text
app.py
features_final.csv
models/ids_bundle.pkl
templates/index.html
```

Start the application:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

### Analysing a CSV file

Upload a packet-level CSV containing:

```text
Time,Source,Destination,Protocol,Length,Info
```

The application will:

1. Read and clean the CSV.
2. Extract protocol values, ports, TCP flags, flow identifiers, and inter-arrival times.
3. Generate 100-packet sliding windows with a 10-packet step.
4. Calculate the 31 model features.
5. Apply the selected classifier.
6. Stream predictions to the dashboard.

---

## Running the SDN Topology

The surviving topology implementation is located at:

```text
sdn/topology/topo.py
```

It implements:

- One remote controller: `c0`
- Five OpenFlow 1.3 switches: `s1` to `s5`
- `s1` as a core switch
- `s2` to `s5` as edge switches
- Twenty hosts
- Five hosts connected to each edge switch
- An HTTP server on `h1`
- `h18`, `h19`, and `h20` assigned attacker roles

Start a basic Ryu OpenFlow 1.3 controller:

```bash
ryu-manager ryu.app.simple_switch_13
```

In another terminal:

```bash
sudo python3 sdn/topology/topo.py
```

Inside the Mininet CLI:

```bash
pingall
```

### Topology documentation note

The archived thesis may describe an earlier linear five-switch topology. This README documents the surviving `topo.py` implementation included in the repository.

---

## Controller Files

The repository contains two different Ryu controller branches.

### Legacy flow-statistics collector

```text
legacy/flow-statistics-collector/Controller_Benign_Traffic.py
```

This controller:

- Requests OpenFlow flow statistics every 10 seconds
- Extracts counters and traffic rates
- Writes results to `FlowStatsfile.csv`
- Assigns benign label `0`
- Does not load a machine-learning model
- Does not install mitigation rules

The surviving file depends on a local `switch.py` module that is not currently included.

### Experimental online IDS controller

```text
experimental/online-ids-controller/traffic_controller.py
```

This experimental branch is designed to provide:

- MAC-learning switching
- ARP spoofing detection
- Online feature extraction
- Random-Forest inference
- OpenFlow drop-rule mitigation

The surviving file depends on additional modules and model files that are not currently included:

```text
feature_extractor.py
ml_engine.py
mitigation.py
arp_guard.py
models/rf_model.pkl
models/scaler.pkl
```

This branch was not part of the validated final CSV-based dashboard workflow and should not be treated as the final deployed system.

---

## Known Limitations

- The final dashboard analyses recorded CSV files rather than receiving live packets from the Ryu controller.
- Automatic OpenFlow mitigation was not validated as part of the final dashboard workflow.
- The experimental online controller is incomplete because supporting modules and model files are unavailable.
- The legacy statistics collector depends on a missing `switch.py` file.
- The current Flask server is a development server intended for local use.
- The model bundle may require compatible versions of scikit-learn, XGBoost, and joblib.
- Original large raw packet captures are not included.
- The archived LaTeX source is incomplete unless all referenced figures and the bibliography are recovered.
- Some dashboard evaluation values are preserved from experimental results rather than recalculated dynamically on every run.

---

## Security Notice

This application is intended for trusted local use.

Python pickle and joblib files can execute code while being loaded. Never load a model file from an untrusted source.

The current dashboard:

- Accepts uploaded CSV files
- Supports uploaded pickle and joblib models
- Uses a development Flask server
- Does not include authentication
- Is not hardened for public Internet deployment

Do not expose the current application directly to the public Internet.

---

## Project Team

| Name | GitHub | Main Contribution |
|---|---|---|
| Shahd Adel Ahmed | `@USERNAME` | To be completed |
| Aliaa Ahmed Atef | `@USERNAME` | To be completed |
| Ali Eldin Tarek Ali | `@USERNAME` | To be completed |
| Omnia Adel Ahmed | `@USERNAME` | To be completed |
| Omar Hesham Ebrahiem | `@USERNAME` | To be completed |

### Supervisor

**Dr. Samar Ashraf**

Detailed contribution information will be maintained in [`AUTHORS.md`](AUTHORS.md).

---

## Repository History

The project was developed before being migrated to GitHub.

The commit history will be reconstructed from dated source files, project backups, experiment records, documentation revisions, and confirmed team milestones. Historical commit dates are intended to represent the approximate dates on which the corresponding components were developed.

See [`HISTORY.md`](HISTORY.md) for further information.

---

## Academic Documentation

The compiled thesis is stored at:

```text
docs/thesis/main.pdf
```

The archived LaTeX chapters are stored under:

```text
docs/thesis/source-incomplete/
```

The source archive may not compile until the original bibliography and all referenced figures are recovered.

---

## Citation

A formal citation entry can be added after the repository name, release version, and publication details are finalized.

```bibtex
@misc{machine_learning_based_ids_2026,
  title        = {Machine Learning-Based Intrusion Detection},
  author       = {Project Team},
  year         = {2026},
  howpublished = {GitHub repository},
  note         = {Graduation project}
}
```

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.
