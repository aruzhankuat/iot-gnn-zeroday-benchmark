# Methodology Passport Card (Task 2)

## 1. Project Information

| Field | Specification |
| :--- | :--- |
| **Research Title** | **Zero-Day Attack Detection in IoT Networks via Graph Neural Networks: From Classical Architectures to Quantum and Quantized Models** |
| **Authors (Student Pair)** | **Kutmyrza Aziza, Kuat Aruzhan** |
| **GitHub Repository** | `https://github.com/aruzhankuat/iot-gnn-zeroday-benchmark` |
| **Artifact Status** | Executable & Verified (W5-W6 Skeleton) \| Dockerfile, config.yaml, src/, data/sample/ |

---

## 2. Research Type

**Quantitative (Experimental, Comparative).**  
All empirical conclusions are derived from measurable numerical metrics (detection accuracy, inference latency in milliseconds, memory consumption in megabytes) recorded under a controlled computational experiment with varied independent factors (model architecture / data representation). Findings are validated using non-parametric statistical hypothesis testing (paired Wilcoxon signed-rank test, significance level $\alpha = 0.05$).  
The qualitative component (literature review from Task 1, Zotero database, ResearchRabbit LitMap) is utilized strictly to provide theoretical grounding for hypothesis thresholds and to select fair baseline models.

---

## 3. Hypotheses Formulation (Refined for Week 5)

**Zero-Day Validation Protocol:** `leave-one-attack-out`.  
Each attack family in the dataset is sequentially withheld from the training phase and introduced exclusively during test evaluation as an unstudied (unseen) zero-day intrusion.

### Hypothesis 1 (RQ1: Quantization & Edge Efficiency)
* **$H_0$ (Null Hypothesis):** INT8-quantized GNN does not reduce median inference latency by $\ge 30\%$ relative to full-precision FP32 GNN, or achieves this latency reduction at the cost of unseen attack precision dropping below $0.92$ / Macro-F1 score dropping by more than $0.02$ (absolute).
* **$H_1$ (Alternative Hypothesis):** INT8-quantized GNN reduces median inference latency by **$\ge 30\%$** and peak memory / model size by **$\ge 40\%$** relative to FP32 GNN, while maintaining precision on unseen attacks **$\ge 0.92$** and Macro-F1 degradation **$\le 0.02$** (absolute) ($p < 0.05$, Wilcoxon signed-rank test).

> **Threshold Rationale (H1):** In the Task 1 literature review, state-of-the-art benchmarks demonstrated a $-38\%$ latency reduction (quantized GNN, IEEE 11424563) and a $-45\%$ memory saving (lightweight Edge-GNN, IEEE 11219271). A $30\%$ threshold establishes a reproducible, conservative margin accounting for runtime overheads. The preliminary "up to 40%" phrasing from Task 1 is replaced with a rigorous one-sided lower bound.  
> **Quantum Model Status (`qgnn`):** The quantum model executes on a classical statevector simulator; its simulated latency and memory are not directly comparable to physical edge micro-nodes. It is incorporated as an **exploratory** comparative baseline (descriptive statistics) rather than as a confirmatory component of H1.

---

### Hypothesis 2 (RQ2: Topological Generalization Advantage)
* **$H_0$ (Null Hypothesis):** GNN models operating on flow interaction graphs do not provide a statistically significant Macro-F1 gain on unseen attacks $\ge 0.15$ over non-graph flat feature baselines, or the observed gain is statistically insignificant ($p \ge 0.05$).
* **$H_1$ (Alternative Hypothesis):** GNN models operating on flow interaction graphs improve Macro-F1 score on unseen attacks by **at least $0.15$** (absolute) over flat feature baselines with statistical significance (Wilcoxon signed-rank test, $p < 0.05$), while maintaining a False Positive Rate no higher than baseline ($\text{FPR}_{\text{GNN}} \le \text{FPR}_{\text{flat}}$).

---

## 4. Variable Matrix

| Category | Variable | Specification / Range / Units |
| :--- | :--- | :--- |
| **Independent Variables** | Model Architecture (H1) | `gnn_fp32` (FP32 baseline), `gnn_int8` (INT8 quantized), `qgnn` (exploratory simulator stub) |
| | Data Representation (H2) | Flow interaction graph topology vs. Flat netflow feature vectors |
| | Held-out Attack Family | Sequentially each attack family in the dataset (leave-one-attack-out) |
| | Message Passing Depth | $k = 2$ hops (fixed in `config.yaml`, ablation scheduled for Week 7+) |
| **Dependent Variables** | Median Inference Latency | Measured execution time (ms per batch and ms per individual flow) |
| | Peak Memory / Model Footprint | Peak RAM allocation (RSS, MB) and serialized disk footprint (MB) |
| | Zero-Day Detection Rate | Recall on held-out unseen attack family ($0.0 \text{ -- } 1.0$) |
| | Classification Performance | Precision, Macro-F1, Accuracy, False Positive Rate (FPR) ($0.0 \text{ -- } 1.0$) |
| **Controlled Variables** | Hardware Constraints | Docker container limits: `--cpus=1 --memory=512m` (edge node emulation) |
| | Software Environment | `python:3.11.9-slim-bookworm`; pinned: `numpy==1.26.4`, `pandas==2.2.2`, `scikit-learn==1.5.1`, `PyYAML==6.0.1` |
| | Dataset & Partitioning | Standardized BoT-IoT sample; benign: 60% chronological train / 40% test; `StandardScaler` fitted strictly on train |
| | Graph Window Size | `window_size = 1000` flows per window (bounds adjacency matrix complexity $O(n^2)$) |
| | Measurement Reproducibility | Fixed `seed = 42`, 3 warmup runs, 30 repeated timing iterations |

---

## 5. Selected Metrics and Baselines

### Metrics Framework
1. **Primary Metrics:**
   * **H1:** Median inference latency ($\text{ms}$).
   * **H2:** Macro-F1 score evaluated strictly on unseen zero-day attack families.
2. **Guardrail Metrics:**
   * Unseen attack precision $\ge 0.92$.
   * Quantization F1 degradation relative to FP32: $\Delta \text{F1} \le 0.02$ (absolute).
   * False Positive Rate constraint: $\text{FPR}_{\text{GNN}} \le \text{FPR}_{\text{flat}}$.
   * Runtime stability: `status == ok` across all active production models.
3. **Secondary Metrics:**
   * Peak RAM consumption (MB), serialized model footprint (MB), Detection Rate (Recall), Accuracy.

### Baseline Models
* **For H1 (Quantization Effect):** Classical **FP32 GNN** evaluated on identical graph topology and hyperparameters (providing a fair comparison isolating computational precision).
* **For H2 (Topological Representation Effect):** **Flat Baseline** evaluated on identical netflow features without relational edge information (Logistic Regression for Weeks 5–6 skeleton; expanded to Random Forest and MLP in Week 7).

---

## 6. Assumptions and Limitations

1. **Graph Construction Topology:** Communication graphs are constructed under the scheme "flows = nodes, edges connect consecutive flows originating from or destined for the same host IP". Sensitivity to topological construction heuristics was identified in Task 1 as an inherent research consideration.
2. **Iterative Engineering Progression:**
   * *Weeks 5–6:* The prototype harness implements parameter-free normalized neighbor propagation with linear classification heads to validate end-to-end benchmarking.
   * *Week 7:* Full-scale PyTorch Geometric architectures (GCN / GraphSAGE) and Quantization-Aware Training (QAT).
3. **Verification Sample Data:** Metrics produced on `data/sample/flows_sample.csv` ($10$ rows) serve exclusively as a functional smoke test verifying pipeline mechanics.
4. **Task 1 Consistency Alignment:** References to "energy efficiency / consumption" in the preliminary Task 1 Bridge Statement have been **strictly removed**, ensuring total alignment with the experimental variables and FINER research questions.

---

## 7. Experiment Configuration (`config.yaml`)

```yaml
# All experiment parameters live here. Edit this file, not the code.
experiment:
  name: sample_smoke_run
  seed: 42

data:
  path: data/sample/flows_sample.csv        # Week 7: switch to full BoT-IoT subset (not committed)
  feature_cols: [duration, pkts, bytes, mean_iat]
  label_col: label
  attack_col: attack_type
  benign_train_frac: 0.6                    # chronological split of benign flows

graph:
  hops: 2                                   # message-passing depth (independent variable)
  window_size: 1000                         # flows per graph window (bounds O(n^2) adjacency)

models:                                     # independent variable: model architecture
  - {name: flat_lr,      enabled: true}     # non-graph baseline (H2)
  - {name: gnn_fp32,     enabled: true}     # classical GNN-style baseline (H1/H2)
  - {name: gnn_int8,     enabled: true}     # quantized GNN (H1)
  - {name: qgnn,         enabled: true}     # quantum-enhanced GNN: stub until Week 7

benchmark:
  warmup_runs: 3
  repeats: 30
  timeout_sec: 600

guardrails:
  min_precision: 0.92                       # from H1
  max_f1_drop_vs_fp32: 0.02

output:
  results_csv: results/results.csv
  log_file: results/run.log
```
