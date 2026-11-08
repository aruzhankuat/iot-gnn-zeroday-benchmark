"""
Collect raw latency / error measurements for Task 3 -> data/raw/test_run.jsonl.

Uses the REAL harness code of this repo (graph_builder, models, data_loader) and
times the same inference callable as src/benchmark.py. The only synthetic part is the
flow table: the full BoT-IoT subset is not committed, so a seeded generator that mimics
the feature scales of data/sample/flows_sample.csv is used. Latency numbers are real
measurements on the machine that runs this script; classification metrics are NOT
scientific results (synthetic flows).

Independent variable (load): number of flows scored per inference call.
Each (model, load, fold) configuration is executed WARMUP + REPEATS times; every call is
written as one JSON line with phase = "warmup" | "eval".

Usage:  python src/collect_runs.py [--out data/raw/test_run.jsonl]
"""
import os

# Same thread limits as the Dockerfile (edge emulation). Must be set BEFORE importing numpy.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
import platform
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import split_leave_one_attack_out  # noqa: E402
from graph_builder import build_host_adjacency, propagate_features  # noqa: E402
from models import FlatLRModel, GNNFP32Model, GNNINT8Model  # noqa: E402

SEED = 42
LOADS = [250, 500, 1000, 2000, 4000, 8000]   # flows per inference call
WARMUP = 5
REPEATS = 30
HOPS = 2
WINDOW = 1000
N_FLOWS = 60000
FOLDS = ["dos_syn", "mirai_botnet", "port_scan"]
CONFIG = {
    "data": {
        "feature_cols": ["duration", "pkts", "bytes", "mean_iat"],
        "label_col": "label",
        "attack_col": "attack_type",
        "benign_train_frac": 0.6,
    }
}


def make_flows(n: int, seed: int) -> pd.DataFrame:
    """Seeded synthetic flow stream (chronological) with partially overlapping classes."""
    rng = np.random.default_rng(seed)
    benign_hosts = [f"192.168.1.{i}" for i in range(10, 40)]
    profiles = {  # (duration, pkts, bytes, mean_iat) medians, same order of magnitude as the sample file
        "benign": (0.3, 15, 1500, 0.02),
        "dos_syn": (0.02, 250, 15000, 0.0001),
        "mirai_botnet": (0.04, 55, 5000, 0.0006),
        "port_scan": (0.01, 3, 180, 0.0008),
    }
    host_of = {"dos_syn": ["192.168.1.100", "192.168.1.101"],
               "mirai_botnet": ["192.168.1.105", "192.168.1.106", "192.168.1.107"],
               "port_scan": ["192.168.1.120"]}
    kinds = rng.choice(list(profiles), size=n, p=[0.70, 0.10, 0.10, 0.10])
    # make attacks bursty (consecutive flows) as in real traces
    for i in range(1, n):
        if kinds[i - 1] != "benign" and rng.random() < 0.6:
            kinds[i] = kinds[i - 1]
    rows = []
    for i, k in enumerate(kinds):
        med = np.array(profiles[k])
        # 8 % of attack flows are "stealthy" (benign-like) -> overlap between classes
        if k != "benign" and rng.random() < 0.08:
            med = np.array(profiles["benign"])
        v = med * rng.lognormal(0.0, 0.35, size=4)
        host = rng.choice(benign_hosts) if k == "benign" else rng.choice(host_of[k])
        rows.append((i + 1, host, v[0], max(int(v[1]), 1), max(int(v[2]), 1), v[3], k, int(k != "benign")))
    return pd.DataFrame(rows, columns=["flow_id", "host_ip", "duration", "pkts", "bytes",
                                       "mean_iat", "attack_type", "label"])


def main(out_path: str):
    np.random.seed(SEED)
    df = make_flows(N_FLOWS, SEED)
    models = {"flat_lr": FlatLRModel, "gnn_fp32": GNNFP32Model, "gnn_int8": GNNINT8Model}
    meta = {"python": platform.python_version(), "numpy": np.__version__,
            "machine": platform.machine(), "cpu_count": os.cpu_count(),
            "omp_threads": os.environ["OMP_NUM_THREADS"], "seed": SEED, "data": "synthetic"}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    n_written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for held_out in FOLDS:
            s = split_leave_one_attack_out(df, CONFIG, held_out)
            blocks_tr = build_host_adjacency(s["train_df"], window_size=WINDOW)
            H_train = propagate_features(s["X_train"], blocks_tr, HOPS)
            for name, cls in models.items():
                m = cls(seed=SEED)
                m.fit(s["X_train"] if name == "flat_lr" else H_train, s["y_train"])
                for load in LOADS:
                    # slice starts where the benign test part begins (before it the test stream
                    # holds only the unseen attack) so every load level contains both classes
                    start = int(np.argmax(s["test_attacks"] == "benign"))
                    assert start + load <= len(s["X_test"]), "test stream shorter than max load"
                    X = s["X_test"][start:start + load]
                    y = s["y_test"][start:start + load]
                    blocks_te = build_host_adjacency(s["test_df"].iloc[start:start + load], window_size=WINDOW)
                    if name == "flat_lr":
                        infer = lambda: m.predict(X)           # noqa: E731
                    else:
                        infer = lambda: m.predict(propagate_features(X, blocks_te, HOPS))  # noqa: E731
                    for run in range(WARMUP + REPEATS):
                        t0 = time.perf_counter()
                        pred = infer()
                        lat_ms = (time.perf_counter() - t0) * 1000.0
                        n_err = int(np.sum(pred != y))
                        benign = y == 0
                        rec = {
                            "model": name, "fold": held_out, "load_flows": load, "run": run,
                            "phase": "warmup" if run < WARMUP else "eval",
                            "latency_ms": round(lat_ms, 5),
                            "throughput_flows_s": round(load / (lat_ms / 1000.0), 2),
                            "n_errors": n_err, "error_rate": round(n_err / load, 6),
                            "fpr": round(float(np.mean(pred[benign] == 1)), 6) if benign.any() else None,
                            "precision": round(float(np.sum((pred == 1) & (y == 1)) / max(np.sum(pred == 1), 1)), 6),
                            "recall": round(float(np.mean(pred[y == 1] == 1)), 6) if (y == 1).any() else None,
                        }
                        if n_written == 0:
                            rec["meta"] = meta
                        f.write(json.dumps(rec) + "\n")
                        n_written += 1
    print(f"wrote {n_written} records -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/test_run.jsonl")
    main(ap.parse_args().out)
