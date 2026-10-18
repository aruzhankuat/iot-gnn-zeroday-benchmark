"""
Main execution benchmark script for IoT GNN Zero-Day Detection.
Implements warmup, repeated latency measurements, memory tracking,
and metric calculation with guardrail validation against H1 & H2.
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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
    """Measure inference latency (median ms) and peak memory consumption (MB)."""
    # Warmup runs
    for _ in range(warmup):
        _ = model.predict(X_eval)

    # Latency repeats
    durations_ms = []
    tracemalloc.start()
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = model.predict(X_eval)
        t1 = time.perf_counter()
        durations_ms.append((t1 - t0) * 1000.0)

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    median_lat = float(np.median(durations_ms))
    p95_lat = float(np.percentile(durations_ms, 95))
    peak_mem_mb = float(peak_mem / (1024 * 1024))
    per_flow_lat = median_lat / max(len(X_eval), 1)

    return {
        "median_latency_ms": round(median_lat, 4),
        "p95_latency_ms": round(p95_lat, 4),
        "latency_per_flow_ms": round(per_flow_lat, 6),
        "peak_memory_mb": round(peak_mem_mb, 4),
    }


def evaluate_predictions(y_true, y_pred, test_attacks, held_out_attack: str):
    """Calculate detection metrics, zero-day recall, and FPR."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # False Positive Rate (FPR = FP / (FP + TN)) on benign test traffic
    benign_mask = test_attacks == "benign"
    if np.sum(benign_mask) > 0:
        tn = np.sum((y_true[benign_mask] == 0) & (y_pred[benign_mask] == 0))
        fp = np.sum((y_true[benign_mask] == 0) & (y_pred[benign_mask] == 1))
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    else:
        fpr = 0.0

    # Zero-Day Detection Rate (Recall strictly on held-out attack family)
    unseen_mask = test_attacks == held_out_attack
    if np.sum(unseen_mask) > 0:
        unseen_rec = recall_score(y_true[unseen_mask], y_pred[unseen_mask], zero_division=0)
        unseen_prec = precision_score(y_true[unseen_mask], y_pred[unseen_mask], zero_division=0)
    else:
        unseen_rec = rec
        unseen_prec = prec

    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "fpr": round(float(fpr), 4),
        "unseen_recall_dr": round(float(unseen_rec), 4),
        "unseen_precision": round(float(unseen_prec), 4),
    }


def run_benchmark(config_path: str = "config.yaml"):
    """Main benchmark runner function."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger = setup_logger(config["output"]["log_file"])
    logger.info("=== Starting IoT GNN Zero-Day Benchmark (Weeks 5-6 Skeleton) ===")
    logger.info(f"Loaded configuration from: {config_path}")
    logger.info(f"Experiment Name: {config['experiment']['name']}, Seed: {config['experiment']['seed']}")

    # 1. Load data
    df = load_dataset(config)
    logger.info(f"Loaded dataset: {len(df)} total flows")

    # 2. Split Leave-One-Attack-Out
    splits = split_leave_one_attack_out(df, config)
    held_out = splits["held_out_attack"]
    logger.info(f"Zero-Day Held-out attack family: '{held_out}'")
    logger.info(f"Train set: {len(splits['X_train'])} flows, Test set: {len(splits['X_test'])} flows")

    # 3. Construct Graph & Propagate
    hops = config["graph"].get("hops", 2)
    win_size = config["graph"].get("window_size", 1000)
    logger.info(f"Constructing host graph: hops={hops}, window_size={win_size}")

    A_train = build_host_adjacency(splits["train_df"], window_size=win_size)
    H_train = propagate_features(splits["X_train"], A_train, hops=hops)

    A_test = build_host_adjacency(splits["test_df"], window_size=win_size)
    H_test = propagate_features(splits["X_test"], A_test, hops=hops)

    # 4. Model Registry
    enabled_models = [m["name"] for m in config["models"] if m.get("enabled", True)]
    warmup = config["benchmark"].get("warmup_runs", 3)
    repeats = config["benchmark"].get("repeats", 30)
    min_prec_threshold = config["guardrails"].get("min_precision", 0.92)
    max_f1_drop_threshold = config["guardrails"].get("max_f1_drop_vs_fp32", 0.02)

    results = []
    fp32_f1 = None

    for model_name in enabled_models:
        logger.info(f"--- Running Benchmark for Model: [{model_name}] ---")
        status = "ok"

        try:
            if model_name == "flat_lr":
                m = FlatLRModel(seed=config["experiment"]["seed"])
                m.fit(splits["X_train"], splits["y_train"])
                bench = measure_latency_and_memory(m, splits["X_test"], warmup, repeats)
                preds = m.predict(splits["X_test"])
            elif model_name == "gnn_fp32":
                m = GNNFP32Model(seed=config["experiment"]["seed"])
                m.fit(H_train, splits["y_train"])
                bench = measure_latency_and_memory(m, H_test, warmup, repeats)
                preds = m.predict(H_test)
            elif model_name == "gnn_int8":
                m = GNNINT8Model(seed=config["experiment"]["seed"])
                m.fit(H_train, splits["y_train"])
                bench = measure_latency_and_memory(m, H_test, warmup, repeats)
                preds = m.predict(H_test)
            elif model_name == "qgnn":
                m = QGNNStubModel(seed=config["experiment"]["seed"])
                m.fit(H_train, splits["y_train"])
                bench = measure_latency_and_memory(m, H_test, warmup, repeats)
                preds = m.predict(H_test)
                status = "exploratory_stub"
            else:
                logger.warning(f"Unknown model architecture: {model_name}, skipping.")
                continue

            metrics = evaluate_predictions(
                splits["y_test"], preds, splits["test_attacks"], held_out
            )

            # Record FP32 baseline F1 for H1 guardrail comparison
            if model_name == "gnn_fp32":
                fp32_f1 = metrics["f1"]

            # Guardrail evaluation
            guardrail_passed = True
            guardrail_note = "pass"

            if model_name == "gnn_int8" and fp32_f1 is not None:
                f1_drop = fp32_f1 - metrics["f1"]
                if metrics["unseen_precision"] < min_prec_threshold:
                    guardrail_passed = False
                    guardrail_note = f"FAIL: precision {metrics['unseen_precision']} < {min_prec_threshold}"
                elif f1_drop > max_f1_drop_threshold:
                    guardrail_passed = False
                    guardrail_note = f"FAIL: F1 drop {round(f1_drop, 4)} > {max_f1_drop_threshold}"

            row = {
                "model": model_name,
                "status": status,
                "held_out_attack": held_out,
                "median_latency_ms": bench["median_latency_ms"],
                "latency_per_flow_ms": bench["latency_per_flow_ms"],
                "peak_memory_mb": bench["peak_memory_mb"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "fpr": metrics["fpr"],
                "unseen_recall_dr": metrics["unseen_recall_dr"],
                "unseen_precision": metrics["unseen_precision"],
                "guardrail_status": guardrail_note,
            }
            results.append(row)
            logger.info(
                f"[{model_name}] Latency: {bench['median_latency_ms']} ms | Memory: {bench['peak_memory_mb']} MB | "
                f"F1: {metrics['f1']} | Unseen DR: {metrics['unseen_recall_dr']} | Guardrail: {guardrail_note}"
            )

        except Exception as e:
            logger.error(f"Error executing benchmark on {model_name}: {str(e)}", exc_info=True)
            results.append({
                "model": model_name,
                "status": f"error: {str(e)}",
                "held_out_attack": held_out,
                "median_latency_ms": None,
                "latency_per_flow_ms": None,
                "peak_memory_mb": None,
                "accuracy": None,
                "precision": None,
                "recall": None,
                "f1": None,
                "fpr": None,
                "unseen_recall_dr": None,
                "unseen_precision": None,
                "guardrail_status": "error",
            })

    # 5. Export results
    results_df = pd.DataFrame(results)
    out_csv = config["output"]["results_csv"]
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    results_df.to_csv(out_csv, index=False)
    logger.info(f"Benchmark results successfully written to: {out_csv}")

    # Print summary table
    logger.info("\n=== BENCHMARK SUMMARY TABLE ===")
    print(results_df[["model", "status", "median_latency_ms", "peak_memory_mb", "f1", "unseen_recall_dr", "guardrail_status"]].to_string(index=False))
    logger.info("=== Execution Completed Successfully ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IoT GNN Zero-Day Benchmark Runner")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    run_benchmark(config_path=args.config)
