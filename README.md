# Zero-Day Attack Detection in IoT Networks via Graph Neural Networks: From Classical Architectures to Quantum and Quantized Models

[![Python 3.11](https://img.shields.io/badge/python-3.11.9-blue.svg)](https://www.python.org/downloads/release/python-3119/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Edge Emulation](https://img.shields.io/badge/Docker-1%20CPU%20%7C%20512MB-green.svg)](Dockerfile)

An empirical benchmarking framework evaluating classical, quantized (`INT8`), and exploratory quantum-enhanced (`QGNN`) Graph Neural Network architectures for real-time zero-day intrusion detection in edge-constrained IoT environments.

> **Status (Weeks 5-6 skeleton).** The models are parameter-free graph propagation followed by a linear head, so the harness runs with a minimal, exact-pinned stack. The quantum model is a stub (reported as `skipped`). Trained PyTorch Geometric GCN/GraphSAGE, INT8 quantization-aware training and a quantum circuit are planned for Week 7. Metrics produced on `data/sample/` only verify that the pipeline works; they are **not** scientific results.

---

## Authors & Affiliation
* **Kutmyrza Aziza** — [GitHub](https://github.com/aziza-ops)
* **Kuat Aruzhan** — [GitHub](https://github.com/aruzhankuat)

---

## 1. Research Overview & Questions

Internet of Things (IoT) ecosystems face increasingly sophisticated, non-stationary cyber threats. Traditional flat-feature detectors struggle against unstudied zero-day exploits whose statistical signatures mutate. While Graph Neural Networks (GNNs) capture relational communication topologies, full-precision models suffer high memory footprint and execution latency on resource-constrained edge nodes.

This research addresses two core FINER research questions:
* **RQ1 (Optimization & Edge Efficiency):** *How do quantized (INT8) and quantum-enhanced GNN architectures compare to classical FP32 GNNs in terms of execution latency and memory footprint when deployed on edge IoT nodes?*
* **RQ2 (Causal Detection & Generalization):** *To what extent does graph topological encoding improve zero-day (unseen) intrusion detection precision and lower false alarm rates compared to non-graph baseline detectors?*

### Testable Hypotheses
* **H1 (Quantization on Edge):** Under a strict leave-one-attack-out zero-day protocol, the INT8 quantized GNN reduces median inference latency by at least 30 % and model size / peak memory by at least 40 % compared to the FP32 GNN, while keeping precision on the test set >= 0.92 and F1 degradation <= 0.02 (absolute). The 30 % threshold is checked with the upper limit of a 95 % bootstrap confidence interval of the latency ratio INT8/FP32 (must be below 0.70); F1 non-inferiority uses a paired Wilcoxon signed-rank test (p < 0.05). *The quantum GNN is evaluated on a simulator as an exploratory baseline only.*
* **H2 (Topological Advantage):** Graph message passing improves F1 (attack class) on unseen zero-day attacks by at least 0.15 (absolute) over flat-feature baselines, with statistical significance (Wilcoxon signed-rank, p < 0.05) and without increasing the false positive rate (FPR_GNN <= FPR_flat).

Replicates for the tests are (attack family x data subsample) pairs; at least 6 pairs are required (subsample seeds 0-4).

---

## 2. Repository Directory Layout

```
iot-gnn-zeroday-benchmark/
├── .gitignore              # No caches, venvs, outputs or heavy raw dumps
├── Dockerfile              # Controlled edge environment (python:3.11.9-slim; limits applied at docker run)
├── LICENSE                 # MIT Open-Source License
├── README.md               # Academic documentation & quickstart
├── requirements.txt        # Exact pinned dependencies (incl. transitive)
├── config.yaml             # Centralized experiment parameters (edit here, not in code)
├── data/
│   └── sample/
│       └── flows_sample.csv # 10-row micro dataset for pipeline smoke verification
├── docs/
│   └── METHODOLOGY_PASSPORT_EN.md # Methodology Passport Card
├── results/
│   ├── .gitkeep
│   ├── results.csv         # Generated benchmark metrics (git-ignored)
│   └── run.log             # Generated execution log (git-ignored)
└── src/
    ├── __init__.py
    ├── benchmark.py        # Main harness (leave-one-attack-out loop, warmup, repeats, timing, metrics, guardrails)
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
| `gnn_int8` | Symmetric INT8 Quantized Head | Edge Compression Target (H1) | Docker (1 CPU, 512MB RAM) | **Active** | PyTorch QAT / ONNX INT8 |
| `qgnn` | Parameterized Quantum Circuit | Exploratory Baseline | Statevector Simulator | **Stub (skipped)** | PennyLane / Qiskit Hybrid PQC |

---

## 4. System Requirements

* **Operating System:** Linux, macOS, or Windows (WSL2 recommended)
* **Python Runtime:** Python `3.11.9`
* **Docker Engine:** Version 20.10+ (for edge container emulation)
* **Edge Resource Emulation:** Hard limits applied at run time: `--cpus=1 --memory=512m`

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

# Optional: keep results on your machine
docker run --rm --cpus=1 --memory=512m -v "${PWD}/results:/app/results" iot-gnn-edge
```

---

## 6. Output Artifacts

Execution outputs are written to `results/` (git-ignored):
* `results/results.csv`: one row per (model, held-out attack family) with columns `model`, `status`, `held_out_attack`, `median_latency_ms`, `p95_latency_ms`, `latency_per_flow_ms`, `peak_memory_mb` (peak Python heap, tracemalloc), `model_size_mb` (parameter size), `accuracy`, `precision`, `recall`, `f1` (attack class), `fpr`, `unseen_recall_dr` (zero-day detection rate) and `guardrail_status`.
* `results/run.log`: timestamped trace of every fold (including a check that the held-out attack family is absent from the training set), per-model metrics and guardrail results, followed by a summary table.

---

## 7. Citation & Methodology
Detailed operationalization, variable matrix, and hypotheses are documented in:
👉 [`docs/METHODOLOGY_PASSPORT_EN.md`](docs/METHODOLOGY_PASSPORT_EN.md)

If utilizing this benchmark harness, cite:
```bibtex
@misc{kutmyrza_kuat_2026_gnn,
  author    = {Kutmyrza Aziza and Kuat Aruzhan},
  title     = {Zero-Day Attack Detection in IoT Networks via Graph Neural Networks: From Classical Architectures to Quantum and Quantized Models},
  year      = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/aruzhankuat/iot-gnn-zeroday-benchmark}}
}
```

---

## 8. License
Distributed under the [MIT License](LICENSE).
