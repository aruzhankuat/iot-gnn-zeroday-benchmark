"""
Main execution benchmark script for IoT GNN Zero-Day Detection.
Implements warmup, repeated latency measurements, peak heap tracking (tracemalloc),
model parameter size reporting, and metric calculation with guardrail validation
against H1 & H2.

Leave-one-attack-out (LOAO) loop: every unique attack family is held out once,
producing one result row per (model, fold). This yields >= num_attack_families
paired observations — sufficient for Wilcoxon signed-rank tests across folds.
"""

import argparse
import logging
import os
import sys
import time
import tracemalloc
import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from data_loader import load_dataset, split_leave_one_attack_out
from graph_builder import build_host_adjacency, propagate_features
from models import FlatLRModel, GNNFP32Model, GNNINT8Model, QGNNStubModel


def setup_logger(log_file: str) -> logging.Logger:
    """Configure benchmark file and stream logging."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger("iot_gnn_benchmark")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")

    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def measure_latency_and_memory(model, X_eval, warmup: int = 3, repeats: int = 30):
    """
    Measure inference latency (median ms over `repeats` runs) and
    peak Python heap allocation via tracemalloc (MB).

    Note: tracemalloc measures Python object heap, not OS RSS.
    For model parameter size use model.param_size_mb().
    """
    # Warmup runs to fill CPU caches and trigger JIT paths
    for _ in range(warmup):
        model.predict(X_eval)

    # Timed repeats under tracemalloc
    durations_ms = []
    tracemalloc.start()
    for _ in range(repeats):
        t0 = time.perf_counter()
        model.predict(X_eval)
        t1 = time.perf_counter()
        durations_ms.append((t1 - t0) * 1000.0)

    _current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    median_lat = float(np.median(durations_ms))
    p95_lat = float(np.percentile(durations_ms, 95))
    peak_heap_mb = float(peak_mem / (1024 * 1024))
    per_flow_lat = median_lat / max(len(X_eval), 1)

    return {
        "median_latency_ms": round(median_lat, 4),
        "p95_latency_ms": round(p95_lat, 4),
        "latency_per_flow_ms": round(per_flow_lat, 6),
        "peak_heap_mb": round(peak_heap_mb, 4),  # tracemalloc Python heap, not RSS
    }


def evaluate_predictions(y_true, y_pred, test_attacks, held_out_attack: str):
    """
    Calculate detection metrics:
    - accuracy / precision / recall / F1 (attack class) on full test set
    - FPR on benign-only rows
    - unseen_recall_dr: recall strictly on held-out attack rows
    - test_precision: precision on full test set (benign + unseen) — used for H1 guardrail
    """
    acc = accuracy_score(y_true, y_pred)
    # Binary F1 for attack class (not Macro-F1)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Precision & recall on full test set (benign + unseen attack)
    test_prec = precision_score(y_true, y_pred, zero_division=0)
    test_rec = recall_score(y_true, y_pred, zero_division=0)

    # False Positive Rate on benign rows only
    benign_mask = test_attacks == "benign"
    if np.sum(benign_mask) > 0:
        tn = int(np.sum((y_true[benign_mask] == 0) & (y_pred[benign_mask] == 0)))
        fp = int(np.sum((y_true[benign_mask] == 0) & (y_pred[benign_mask] == 1)))
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    else:
        fpr = 0.0

    # Zero-Day Detection Rate: recall strictly on held-out (unseen) attack family
    unseen_mask = test_attacks == held_out_attack
    if np.sum(unseen_mask) > 0:
        unseen_rec = recall_score(y_true[unseen_mask], y_pred[unseen_mask], zero_division=0)
    else:
        unseen_rec = test_rec

    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(test_prec), 4),   # precision on benign + unseen
        "recall": round(float(test_rec), 4),
        "f1": round(float(f1), 4),                  # binary F1 (attack class)
        "fpr": round(float(fpr), 4),
        "unseen_recall_dr": round(float(unseen_rec), 4),
    }


def run_fold(
    model_name: str,
    splits: dict,
    config: dict,
    logger: logging.Logger,
    fp32_f1_ref: float | None,
):
    """Run one leave-one-attack-out fold for one model. Returns a result dict."""
    hops = config["graph"].get("hops", 2)
    win_size = config["graph"].get("window_size", 1000)
    warmup = config["benchmark"].get("warmup_runs", 3)
    repeats = config["benchmark"].get("repeats", 30)
    min_prec_threshold = config["guardrails"].get("min_precision", 0.92)
    max_f1_drop_threshold = config["guardrails"].get("max_f1_drop_vs_fp32", 0.02)

    held_out = splits["held_out_attack"]

    # Build graphs (sorted by ts; falls back to consecutive if no IP columns)
    A_train = build_host_adjacency(splits["train_df"], window_size=win_size)
    H_train = propagate_features(splits["X_train"], A_train, hops=hops)
    A_test = build_host_adjacency(splits["test_df"], window_size=win_size)
    H_test = propagate_features(splits["X_test"], A_test, hops=hops)

    seed = config["experiment"]["seed"]

    if model_name == "flat_lr":
        m = FlatLRModel(seed=seed)
        m.fit(splits["X_train"], splits["y_train"])
        bench = measure_latency_and_memory(m, splits["X_test"], warmup, repeats)
        preds = m.predict(splits["X_test"])
        status = "ok"

    elif model_name == "gnn_fp32":
        m = GNNFP32Model(seed=seed)
        m.fit(H_train, splits["y_train"])
        bench = measure_latency_and_memory(m, H_test, warmup, repeats)
        preds = m.predict(H_test)
        status = "ok"

    elif model_name == "gnn_int8":
        m = GNNINT8Model(seed=seed)
        m.fit(H_train, splits["y_train"])
        bench = measure_latency_and_memory(m, H_test, warmup, repeats)
        preds = m.predict(H_test)
        status = "ok"

    elif model_name == "qgnn":
        # QGNN is an exploratory stub — record NaN metrics, do NOT pollute results with
        # fabricated numbers from the stub's thresholding heuristic.
        logger.info(f"  [qgnn] Exploratory stub: metrics recorded as NaN (Week 7 PQC pending)")
        return {
            "model": model_name,
            "status": "exploratory_stub",
            "held_out_attack": held_out,
            **{k: None for k in [
                "median_latency_ms", "latency_per_flow_ms", "peak_heap_mb",
                "param_size_mb", "accuracy", "precision", "recall",
                "f1", "fpr", "unseen_recall_dr", "guardrail_status",
            ]},
        }
    else:
        raise ValueError(f"Unknown model: {model_name}")

    metrics = evaluate_predictions(splits["y_test"], preds, splits["test_attacks"], held_out)
    param_mb = m.param_size_mb()
    bench["param_size_mb"] = round(param_mb, 6)

    # Guardrail for H1 (INT8 only)
    guardrail_note = "n/a"
    if model_name == "gnn_int8" and fp32_f1_ref is not None:
        f1_drop = fp32_f1_ref - metrics["f1"]
        # Precision guardrail uses full-test precision (benign + unseen), not unseen-only
        if metrics["precision"] < min_prec_threshold:
            guardrail_note = (
                f"FAIL: test_precision {metrics['precision']} < {min_prec_threshold}"
            )
        elif f1_drop > max_f1_drop_threshold:
            guardrail_note = f"FAIL: F1 drop {round(f1_drop, 4)} > {max_f1_drop_threshold}"
        else:
            guardrail_note = "pass"

    logger.info(
        f"  [{model_name}] fold={held_out} | lat={bench['median_latency_ms']} ms | "
        f"heap={bench['peak_heap_mb']} MB | params={param_mb:.4f} MB | "
        f"F1={metrics['f1']} | DR={metrics['unseen_recall_dr']} | guardrail={guardrail_note}"
    )

    return {
        "model": model_name,
        "status": status,
        "held_out_attack": held_out,
        "median_latency_ms": bench["median_latency_ms"],
        "latency_per_flow_ms": bench["latency_per_flow_ms"],
        "peak_heap_mb": bench["peak_heap_mb"],
        "param_size_mb": bench["param_size_mb"],
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "fpr": metrics["fpr"],
        "unseen_recall_dr": metrics["unseen_recall_dr"],
        "guardrail_status": guardrail_note,
    }


def run_benchmark(config_path: str = "config.yaml"):
    """
    Main benchmark runner.

    Executes a full leave-one-attack-out loop: for each unique attack family in the
    dataset, that family is withheld from training and used as the zero-day test set.
    Each (model, fold) pair produces one row in results.csv, yielding >= K paired
    observations (K = number of attack families) for downstream Wilcoxon tests.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger = setup_logger(config["output"]["log_file"])
    logger.info("=== IoT GNN Zero-Day Benchmark — Executable Skeleton (Weeks 5-6) ===")
    logger.info(f"Config: {config_path} | Experiment: {config['experiment']['name']} | "
                f"Seed: {config['experiment']['seed']}")

    # 1. Load data
    df = load_dataset(config)
    logger.info(f"Dataset loaded: {len(df)} flows")

    # 2. Enumerate unique attack families for LOAO loop
    attack_col = config["data"]["attack_col"]
    attack_families = [a for a in df[attack_col].unique() if a != "benign"]
    if not attack_families:
        raise ValueError("No attack families found in dataset; check 'attack_col' in config.")
    logger.info(f"Attack families for LOAO: {attack_families}")

    enabled_models = [m["name"] for m in config["models"] if m.get("enabled", True)]
    all_results = []

    # 3. Leave-One-Attack-Out loop
    for fold_attack in attack_families:
        logger.info(f"=== LOAO Fold: held_out='{fold_attack}' ===")
        splits = split_leave_one_attack_out(df, config, held_out_attack=fold_attack)
        logger.info(
            f"  Train: {len(splits['X_train'])} flows | Test: {len(splits['X_test'])} flows"
        )

        # Per-fold FP32 baseline F1 needed for INT8 guardrail
        fold_fp32_f1 = None

        for model_name in enabled_models:
            try:
                row = run_fold(model_name, splits, config, logger, fp32_f1_ref=fold_fp32_f1)
                if model_name == "gnn_fp32" and row.get("f1") is not None:
                    fold_fp32_f1 = row["f1"]
                all_results.append(row)
            except Exception as exc:
                logger.error(
                    f"  [{model_name}] fold={fold_attack} FAILED: {exc}", exc_info=True
                )
                all_results.append({
                    "model": model_name,
                    "status": f"error: {exc}",
                    "held_out_attack": fold_attack,
                    **{k: None for k in [
                        "median_latency_ms", "latency_per_flow_ms", "peak_heap_mb",
                        "param_size_mb", "accuracy", "precision", "recall",
                        "f1", "fpr", "unseen_recall_dr",
                    ]},
                    "guardrail_status": "error",
                })

    # 4. Export results
    results_df = pd.DataFrame(all_results)
    out_csv = config["output"]["results_csv"]
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    results_df.to_csv(out_csv, index=False)
    logger.info(f"Results written to: {out_csv}")

    # 5. Print summary
    logger.info("\n=== BENCHMARK SUMMARY TABLE ===")
    summary_cols = [
        "model", "held_out_attack", "status",
        "median_latency_ms", "peak_heap_mb", "param_size_mb",
        "f1", "unseen_recall_dr", "guardrail_status",
    ]
    print(results_df[summary_cols].to_string(index=False))
    logger.info("=== Execution Completed Successfully ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IoT GNN Zero-Day Benchmark Runner")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    run_benchmark(config_path=args.config)
