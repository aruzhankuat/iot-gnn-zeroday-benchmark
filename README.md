# Zero-Day Attack Detection in IoT Networks via Graph Neural Networks: From Classical Architectures to Quantum and Quantized Models

[![Python 3.11](https://img.shields.io/badge/python-3.11.9-blue.svg)](https://www.python.org/downloads/release/python-3119/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Edge Emulation](https://img.shields.io/badge/Docker-1%20CPU%20%7C%20512MB-green.svg)](Dockerfile)

An empirical benchmarking framework evaluating classical, quantized (`INT8`), and exploratory quantum-enhanced (`QGNN`) Graph Neural Network architectures for real-time zero-day intrusion detection in edge-constrained IoT environments.

---

## Authors & Affiliation
* **Kutmyrza Aziza** — [GitHub](https://github.com/aruzhankuat)
* **Kuat Aruzhan** — [GitHub](https://github.com/aruzhankuat)

---

## 1. Research Overview & Questions

Internet of Things (IoT) ecosystems face increasingly sophisticated, non-stationary cyber threats. Traditional flat-feature detectors struggle against unstudied zero-day exploits whose statistical signatures mutate. While Graph Neural Networks (GNNs) capture relational communication topologies, full-precision models suffer high memory footprint and execution latency on resource-constrained edge nodes.

This research addresses two core FINER research questions:
* **RQ1 (Optimization & Edge Efficiency):** *How do quantized (INT8) and quantum-enhanced GNN architectures compare to classical FP32 GNNs in terms of execution latency and memory footprint when deployed on edge IoT nodes?*
* **RQ2 (Causal Detection & Generalization):** *To what extent does graph topological encoding improve zero-day (unseen) intrusion detection precision and lower false alarm rates compared to non-graph baseline detectors?*

### Testable Hypotheses
* **H1 (Quantization on Edge):** Under a strict leave-one-attack-out zero-day protocol, INT8 quantized GNN reduces median inference latency by **$\ge 30\%$** and peak memory / model size by **$\ge 40\%$** compared to FP32 GNN, maintaining zero-day precision **$\ge 0.92$** and F1 drop **$\le 0.02$** ($p < 0.05$, Wilcoxon signed-rank test). *Quantum GNN is evaluated on simulator as an exploratory baseline.*
* **H2 (Topological Advantage):** Graph topological message passing boosts Macro-F1 on unseen zero-day attacks by **$\ge 0.15$** (absolute) over flat-feature baselines with statistical significance ($p < 0.05$) without increasing False Positive Rate ($\text{FPR}_{\text{GNN}} \le \text{FPR}_{\text{flat}}$).

---

## 2. Repository Directory Layout

```
iot-gnn-zeroday-benchmark/
├── .gitignore              # Cleaned gitignore (no caches, venv, or heavy raw dumps)
├── Dockerfile              # Controlled edge environment (python:3.11.9-slim, 1 CPU, 512MB)
├── LICENSE                 # MIT Open-Source License
├── README.md               # Academic documentation & quickstart
├── requirements.txt        # Exact pinned dependencies
├── config.yaml             # Centralized experiment parameters (edit here, not in code)
├── data/
│   └── sample/
│       └── flows_sample.csv # 10-row micro dataset for pipeline smoke verification
├── docs/
│   └── METHODOLOGY_PASSPORT.md # Official Methodology Passport Card
├── results/
│   ├── .gitkeep
│   ├── results.csv         # Measured benchmark metrics output
│   └── run.log             # Execution and guardrail evaluation log
└── src/
    ├── __init__.py
    ├── benchmark.py        # Main execution harness (warmup, repeats, timing, metrics)
    ├── data_loader.py      # Chronological split & leave-one-attack-out generator
    ├── graph_builder.py    # Host-based temporal graph windowing & k-hop propagation
    └── models.py           # Models: flat_lr, gnn_fp32, gnn_int8, qgnn (exploratory stub)
```

---

## 3. Planned Benchmark Matrix

| Model Identifier | Architecture & Precision | Role in Research | Target Hardware / Environment | Status (W5-W6) | Week 7+ Milestone |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `flat_lr` | Flat Logistic Regression | Non-graph Baseline (H2) | Docker (1 CPU, 512MB RAM) | **Active** | Add Random Forest & MLP |
| `gnn_fp32` | 2-Hop Prop. + FP32 Linear Head | Classical GNN Baseline (H1/H2) | Docker (1 CPU, 512MB RAM) | **Active** | PyG GCN / GraphSAGE FP32 |
| `gnn_int8` | Symmetric INT8 Quantized GNN | Edge Compression Target (H1) | Docker (1 CPU, 512MB RAM) | **Active** | PyTorch QAT / ONNX INT8 |
| `qgnn` | Parameterized Quantum Circuit | Exploratory Baseline | Statevector Simulator | **Stub** | PennyLane / Qiskit Hybrid PQC |

---

## 4. System Requirements

* **Operating System:** Linux, macOS, or Windows (WSL2 recommended)
* **Python Runtime:** Python `3.11.9`
* **Docker Engine:** Version 20.10+ (for edge container emulation)
* **Edge Resource Emulation:** Hard limits: `--cpus=1 --memory=512m`

---

## 5. Quickstart Guide (Run in 2-3 Commands)

### Option A: Local Python Environment
```bash
# 1. Clone repository and install exact pinned dependencies
git clone https://github.com/aruzhankuat/iot-gnn-zeroday-benchmark.git
cd iot-gnn-zeroday-benchmark
pip install -r requirements.txt

# 2. Run benchmark on sample verification data
python src/benchmark.py --config config.yaml
```

### Option B: Docker Edge Emulation (Recommended)
```bash
# 1. Build reproducible container
docker build -t iot-gnn-edge .

# 2. Run under edge hardware constraint (1 CPU core, 512MB RAM)
docker run --rm --cpus=1 --memory=512m iot-gnn-edge
```

---

## 6. Output Artifacts

Execution outputs are recorded in `results/`:
* `results/results.csv`: Contains structured columns for `model`, `median_latency_ms`, `latency_per_flow_ms`, `peak_memory_mb`, `accuracy`, `precision`, `recall`, `f1`, `fpr`, `unseen_recall_dr`, and `guardrail_status`.
* `results/run.log`: Full timestamped execution trace verifying zero data-leakage splits and guardrail thresholds.

---

## 7. Citation & Methodology
Detailed operationalization, variable matrix, and mathematical hypotheses are documented in:
👉 [`docs/METHODOLOGY_PASSPORT.md`](docs/METHODOLOGY_PASSPORT_EN.md)

If utilizing this benchmark harness, cite:
```bibtex
@misc{kutmyrza_kuat_2026_gnn,
  author    = {Kutmyrza, Aziza and Kuat, Aruzhan},
  title     = {Zero-Day Attack Detection in IoT Networks via Graph Neural Networks: From Classical Architectures to Quantum and Quantized Models},
  year      = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/aruzhankuat/iot-gnn-zeroday-benchmark}}
}
```

---

## 8. License
Distributed under the [MIT License](LICENSE).
