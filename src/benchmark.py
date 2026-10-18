"""
Main execution benchmark script for IoT GNN Zero-Day Detection.
Leave-one-attack-out loop over all attack families, with warmup, repeated latency
measurements (graph propagation included for graph models), memory tracking,
and guardrail validation against H1.
"""

import argparse
import logging
import os
import sys
import time
import tracemalloc

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from data_loader import list_attack_families, load_dataset, split_leave_one_attack_out
from graph_builder import build_host_adjacency, propagate_features
from models import FlatLRModel, GNNFP32Model, GNNINT8Model, QGNNStubModel

MODEL_CLASSES = {
    "flat_lr": FlatLRModel,
    "gnn_fp32": GNNFP32Model,
    "gnn_int8": GNNINT8Model,
    "qgnn": QGNNStubModel,
}
COLUMNS = [
    "model", "status", "held_out_attack", "median_latency_ms", "p95_latency_ms",
    "latency_per_flow_ms", "peak_memory_mb", "model_size_mb", "accuracy", "precision",
    "recall", "f1", "fpr", "unseen_recall_dr", "guardrail_status",
]


def setup_logger(log_file: str) -> logging.Logger:
    d = os.path.dirname(log_file)
    if d:
        os.makedirs(d, exist_ok=True)
    logger = logging.getLogger("iot_gnn_benchmark")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    for h in (logging.FileHandler(log_file, mode="w", encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def measure_latency_and_memory(infer, n_flows: int, warmup: int = 3, repeats: int = 30):
    """
    `infer` is the full inference callable (propagation + head for graph models).
    Latency is timed WITHOUT tracemalloc (it adds overhead); memory is measured
    in a separate single call.
    """
    for _ in range(warmup):
        infer()

    durations_ms = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        infer()
        durations_ms.append((time.perf_counter() - t0) * 1000.0)

    tracemalloc.start()
    infer()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    median = float(np.median(durations_ms))
    return {
        "median_latency_ms": round(median, 4),
        "p95_latency_ms": round(float(np.percentile(durations_ms, 95)), 4),
        "latency_per_flow_ms": round(median / max(n_flows, 1), 6),
        "peak_memory_mb": round(peak / (1024 * 1024), 4),
    }


def evaluate_predictions(y_true, y_pred, test_attacks, held_out_attack: str):
    """Detection metrics over the whole test set (benign + unseen attack)."""
    benign = test_attacks == "benign"
    fp = int(np.sum((y_pred[benign] == 1)))
    tn = int(np.sum((y_pred[benign] == 0)))
    unseen = test_attacks == held_out_attack
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "fpr": round(fp / (fp + tn), 4) if (fp + tn) > 0 else 0.0,
        # zero-day detection rate: share of unseen-attack flows that were flagged
        "unseen_recall_dr": round(float(np.mean(y_pred[unseen] == 1)), 4) if unseen.any() else float("nan"),
    }


def run_benchmark(config_path: str = "config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger = setup_logger(config["output"]["log_file"])
    logger.info("=== Starting IoT GNN Zero-Day Benchmark (Weeks 5-6 Skeleton) ===")
    seed = config["experiment"]["seed"]
    np.random.seed(seed)
    logger.info(f"Config: {config_path} | Experiment: {config['experiment']['name']} | Seed: {seed}")

    df = load_dataset(config)
    families = list_attack_families(df, config)
    logger.info(f"Loaded {len(df)} flows; attack families (each held out in turn): {families}")

    hops = config["graph"].get("hops", 2)
    win_size = config["graph"].get("window_size", 1000)
    warmup = config["benchmark"].get("warmup_runs", 3)
    repeats = config["benchmark"].get("repeats", 30)
    min_prec = config["guardrails"].get("min_precision", 0.92)
    max_f1_drop = config["guardrails"].get("max_f1_drop_vs_fp32", 0.02)
    enabled = [m["name"] for m in config["models"] if m.get("enabled", True)]

    results = []
    for held_out in families:
        s = split_leave_one_attack_out(df, config, held_out)
        n_tr, n_te = len(s["X_train"]), len(s["X_test"])
        logger.info(f"--- Fold: held-out '{held_out}' | train={n_tr} test={n_te} "
                    f"(held-out family absent from train: "
                    f"{held_out not in set(s['train_df'][config['data']['attack_col']])}) ---")

        blocks_tr = build_host_adjacency(s["train_df"], window_size=win_size)
        blocks_te = build_host_adjacency(s["test_df"], window_size=win_size)
        H_train = propagate_features(s["X_train"], blocks_tr, hops)
        fp32_f1 = None

        for name in enabled:
            row = {c: None for c in COLUMNS}
            row.update(model=name, held_out_attack=held_out, status="ok", guardrail_status="n/a")
            try:
                if name not in MODEL_CLASSES:
                    raise KeyError(f"unknown model '{name}'")
                m = MODEL_CLASSES[name](seed=seed)
                if len(set(s["y_train"])) < 2:
                    raise ValueError("train split contains a single class")

                if name == "flat_lr":
                    m.fit(s["X_train"], s["y_train"])
                    infer = lambda: m.predict(s["X_test"])
                else:
                    m.fit(H_train, s["y_train"])
                    infer = lambda: m.predict(propagate_features(s["X_test"], blocks_te, hops))

                bench = measure_latency_and_memory(infer, n_te, warmup, repeats)
                metrics = evaluate_predictions(s["y_test"], infer(), s["test_attacks"], held_out)
                row.update(bench)
                row.update(metrics)
                row["model_size_mb"] = round(m.size_mb(), 8)

                if name == "gnn_fp32":
                    fp32_f1 = metrics["f1"]
                if name == "gnn_int8" and fp32_f1 is not None:
                    drop = fp32_f1 - metrics["f1"]
                    if metrics["precision"] < min_prec:
                        row["guardrail_status"] = f"FAIL: precision {metrics['precision']} < {min_prec}"
                    elif drop > max_f1_drop:
                        row["guardrail_status"] = f"FAIL: F1 drop {round(drop, 4)} > {max_f1_drop}"
                    else:
                        row["guardrail_status"] = "pass"
                logger.info(f"[{name}] lat={row['median_latency_ms']} ms | mem={row['peak_memory_mb']} MB | "
                            f"F1={row['f1']} | FPR={row['fpr']} | DR={row['unseen_recall_dr']} | "
                            f"guardrail={row['guardrail_status']}")
            except NotImplementedError as e:
                row["status"] = f"skipped: {e}"
                logger.info(f"[{name}] skipped: {e}")
            except Exception as e:
                row["status"] = f"error: {e}"
                row["guardrail_status"] = "error"
                logger.error(f"[{name}] failed: {e}", exc_info=True)
            results.append(row)

    out_csv = config["output"]["results_csv"]
    d = os.path.dirname(out_csv)
    if d:
        os.makedirs(d, exist_ok=True)
    res = pd.DataFrame(results, columns=COLUMNS)
    res.to_csv(out_csv, index=False)
    logger.info(f"Results written to: {out_csv}")

    ok = res[res["status"] == "ok"]
    if not ok.empty:
        agg = ok.groupby("model")[["median_latency_ms", "model_size_mb", "f1", "fpr", "unseen_recall_dr"]].mean()
        if "gnn_fp32" in agg.index:
            base = agg.loc["gnn_fp32"]
            agg["latency_vs_fp32_%"] = (agg["median_latency_ms"] / base["median_latency_ms"] - 1) * 100
            agg["size_vs_fp32_%"] = (agg["model_size_mb"] / base["model_size_mb"] - 1) * 100
        logger.info("=== SUMMARY (mean over folds) ===")
        print(agg.to_string(float_format=lambda v: f"{v:.6g}"))
    logger.info("=== Execution Completed ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IoT GNN Zero-Day Benchmark Runner")
    parser.add_argument("--config", type=str, default="config.yaml")
    run_benchmark(config_path=parser.parse_args().config)
