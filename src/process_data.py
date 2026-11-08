"""
Data processing and statistical aggregation for Task 3.

Pipeline:  data/raw/test_run.jsonl --(drop warm-up)--> eval runs
           --(aggregate per model x load)--> data/processed/metrics_summary.csv

All resampling uses an explicit numpy Generator with a fixed seed (SEED = 42), so every
number in the notebook / README is reproducible bit-for-bit.

Usage:  python src/process_data.py
"""
import argparse
import os

import numpy as np
import pandas as pd

SEED = 42
N_BOOT = 5000
CI_LEVEL = 0.95
BASELINE = "gnn_fp32"
PROPOSED = "gnn_int8"

# guardrail thresholds taken from config.yaml (H1)
MIN_PRECISION = 0.92
MAX_F1_DROP = 0.02


# ----------------------------------------------------------------------------- ingestion
def load_jsonl(path: str) -> pd.DataFrame:
    """Read the raw benchmark log (one JSON object per line)."""
    df = pd.read_json(path, lines=True)
    if "meta" in df.columns:
        df = df.drop(columns="meta")
    return df


def filter_warmup(df: pd.DataFrame) -> pd.DataFrame:
    """Discard cold-start runs (phase == 'warmup'); keep only evaluation runs."""
    return df[df["phase"] == "eval"].reset_index(drop=True)


# ----------------------------------------------------------------------------- statistics
def bootstrap_ci(x, stat=np.median, n_boot: int = N_BOOT, level: float = CI_LEVEL, seed: int = SEED):
    """Percentile bootstrap CI of `stat(x)`. Returns (lo, hi)."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = stat(x[idx], axis=1)
    a = (1 - level) / 2
    return float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a))


def bootstrap_ratio_ci(num, den, n_boot: int = N_BOOT, level: float = CI_LEVEL, seed: int = SEED):
    """
    Bootstrap CI of median(num)/median(den) (independent resampling of both samples).
    Used for H1: ratio INT8/FP32, upper limit must be < 0.70.
    """
    num, den = np.asarray(num, float), np.asarray(den, float)
    rng = np.random.default_rng(seed)
    n = np.median(num[rng.integers(0, len(num), size=(n_boot, len(num)))], axis=1)
    d = np.median(den[rng.integers(0, len(den), size=(n_boot, len(den)))], axis=1)
    r = n / d
    a = (1 - level) / 2
    point = float(np.median(num) / np.median(den))
    return point, float(np.quantile(r, a)), float(np.quantile(r, 1 - a))


def summarize(x, seed: int = SEED) -> dict:
    """Median, quartiles, p95, p99, mean, SD and bootstrapped 95 % CI of the median."""
    x = np.asarray(x, dtype=float)
    lo, hi = bootstrap_ci(x, np.median, seed=seed)
    return {
        "n": int(len(x)),
        "median": float(np.median(x)),
        "p25": float(np.percentile(x, 25)),
        "p75": float(np.percentile(x, 75)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
        "mean": float(x.mean()),
        "sd": float(x.std(ddof=1)),
        "ci95_lo": lo,
        "ci95_hi": hi,
    }


def aggregate(eval_df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (model, load_flows). Latency statistics are pooled over folds and runs;
    throughput and guardrail columns (error rate, FPR, precision) are means over folds.
    """
    rows = []
    for (model, load), g in eval_df.groupby(["model", "load_flows"], sort=True):
        # distinct deterministic seed per group -> independent but reproducible resamples
        seed = SEED + 1000 * (hash_model(model)) + int(load)
        s = summarize(g["latency_ms"], seed=seed)
        tp = summarize(g["throughput_flows_s"], seed=seed)
        rows.append({
            "model": model, "load_flows": int(load), **{f"lat_{k}": v for k, v in s.items()},
            "thr_median": tp["median"], "thr_ci95_lo": tp["ci95_lo"], "thr_ci95_hi": tp["ci95_hi"],
            "error_rate": g["error_rate"].mean(),
            "fpr": g["fpr"].mean(),
            "precision": g["precision"].mean(),
            "recall": g["recall"].mean(),
        })
    return pd.DataFrame(rows)


def hash_model(name: str) -> int:
    """Stable small integer per model name (python's hash() is salted per process)."""
    return sum((i + 1) * ord(c) for i, c in enumerate(name)) % 997


def latency_ratio_table(eval_df: pd.DataFrame, num=PROPOSED, den=BASELINE) -> pd.DataFrame:
    """Median latency ratio num/den with bootstrapped 95 % CI for each load level (H1 check)."""
    rows = []
    for load, g in eval_df.groupby("load_flows"):
        a = g.loc[g["model"] == num, "latency_ms"]
        b = g.loc[g["model"] == den, "latency_ms"]
        r, lo, hi = bootstrap_ratio_ci(a, b, seed=SEED + int(load))
        rows.append({"load_flows": int(load), "ratio": r, "ci95_lo": lo, "ci95_hi": hi,
                     "h1_latency_ok": hi < 0.70})
    return pd.DataFrame(rows)


def guardrail_table(eval_df: pd.DataFrame) -> pd.DataFrame:
    """
    Guardrail metrics per model (mean over loads and folds): error rate, FPR, precision.
    INT8 is additionally checked against the H1 guardrails (precision >= 0.92 and
    F1 drop vs FP32 <= 0.02).
    """
    d = eval_df.copy()
    denom = (d["precision"] + d["recall"]).to_numpy()
    # F1 is 0 (not NaN) when a run flags no attack at all (precision = recall = 0)
    d["f1"] = np.divide(2 * d["precision"] * d["recall"], denom, out=np.zeros(len(d)), where=denom > 0)
    g = d.groupby("model")[["error_rate", "fpr", "precision", "recall", "f1"]].mean()
    g["f1_drop_vs_fp32"] = g.loc[BASELINE, "f1"] - g["f1"]
    g["guardrail"] = np.where(
        (g["precision"] >= MIN_PRECISION) & (g["f1_drop_vs_fp32"] <= MAX_F1_DROP), "pass", "FAIL")
    g.loc[BASELINE, "guardrail"] = "reference"
    return g.reset_index()


def fmt_median_iqr(row, prefix="lat", digits=3) -> str:
    """'median (p25-p75)' string for README tables."""
    return f"{row[prefix + '_median']:.{digits}f} ({row[prefix + '_p25']:.{digits}f}-{row[prefix + '_p75']:.{digits}f})"


def readme_table(summary: pd.DataFrame, ratio: pd.DataFrame, loads=None) -> str:
    """Markdown table Baseline (FP32) vs Proposed (INT8): median (p25-p75), mean ± SD, p95, p99, ratio [95 % CI]."""
    loads = loads or sorted(summary["load_flows"].unique())
    head = ("| Load (flows/call) | Method | Median (p25–p75), ms | Mean ± SD, ms | p95, ms | p99, ms | "
            "Median 95 % CI, ms | INT8/FP32 ratio [95 % CI] |\n|---:|:---|:---|:---|---:|---:|:---|:---|")
    lines = [head]
    for load in loads:
        r = ratio[ratio["load_flows"] == load].iloc[0]
        for model, label in ((BASELINE, "Baseline (GNN FP32)"), (PROPOSED, "Proposed (GNN INT8)")):
            s = summary[(summary["model"] == model) & (summary["load_flows"] == load)].iloc[0]
            rr = f"{r['ratio']:.2f} [{r['ci95_lo']:.2f}; {r['ci95_hi']:.2f}]" if model == PROPOSED else "–"
            lines.append(f"| {load} | {label} | {fmt_median_iqr(s)} | {s['lat_mean']:.3f} ± {s['lat_sd']:.3f} | "
                         f"{s['lat_p95']:.3f} | {s['lat_p99']:.3f} | [{s['lat_ci95_lo']:.3f}; {s['lat_ci95_hi']:.3f}] | {rr} |")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- CLI
def main(raw: str, out: str):
    raw_df = load_jsonl(raw)
    eval_df = filter_warmup(raw_df)
    summary = aggregate(eval_df)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    summary.to_csv(out, index=False, float_format="%.6g")
    print(f"raw records: {len(raw_df)} | warm-up discarded: {len(raw_df) - len(eval_df)} | "
          f"eval kept: {len(eval_df)}")
    print(f"wrote {len(summary)} rows -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/test_run.jsonl")
    ap.add_argument("--out", default="data/processed/metrics_summary.csv")
    a = ap.parse_args()
    main(a.raw, a.out)
