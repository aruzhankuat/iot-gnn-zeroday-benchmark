"""
Publication-ready figures for Task 3 (3 core rules of scientific visualization):
  1. explicit axis titles with physical units, log scales where data span orders of magnitude;
  2. uncertainty always shown (bootstrapped 95 % CI band / error bars);
  3. colour-blind-safe palette + distinct linestyles, markers and hatches (monochrome-safe).
Every figure is exported as vector .pdf and raster .png at 300 DPI.

Usage:  python src/visualize.py
"""
import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import seaborn as sns

from process_data import bootstrap_ci, filter_warmup, load_jsonl, aggregate

DPI = 300
MODELS = ["flat_lr", "gnn_fp32", "gnn_int8"]
LABELS = {"flat_lr": "Flat LR (non-graph)", "gnn_fp32": "GNN FP32 (baseline)", "gnn_int8": "GNN INT8 (proposed)"}
_PAL = sns.color_palette("colorblind")
COLORS = {"flat_lr": _PAL[2], "gnn_fp32": _PAL[0], "gnn_int8": _PAL[3]}
STYLE = {  # linestyle + marker + hatch differ per model -> readable in grayscale print
    "flat_lr": dict(linestyle=":", marker="s", hatch="//"),
    "gnn_fp32": dict(linestyle="-", marker="o", hatch=""),
    "gnn_int8": dict(linestyle="--", marker="^", hatch="xx"),
}


def set_style():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": DPI, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})


def _plain_log_axis(ax):
    """Readable decimal tick labels (0.1, 0.2, 0.5, 1, 2, 5 ...) instead of 10^n on a log axis."""
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def save_figure(fig, name: str, out_dir: str = "reports/figures"):
    """Save `<name>.pdf` (vector) and `<name>.png` (300 DPI)."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(p, dpi=DPI, bbox_inches="tight")
        paths.append(p)
    return paths


def plot_latency_vs_load(summary: pd.DataFrame):
    """Fig 1: median latency vs load, shaded bootstrapped 95 % CI of the median, log-log axes."""
    set_style()
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for m in MODELS:
        d = summary[summary["model"] == m].sort_values("load_flows")
        st = STYLE[m]
        ax.plot(d["load_flows"], d["lat_median"], color=COLORS[m], linestyle=st["linestyle"],
                marker=st["marker"], markersize=6, linewidth=1.8, label=LABELS[m])
        ax.fill_between(d["load_flows"], d["lat_ci95_lo"], d["lat_ci95_hi"], color=COLORS[m], alpha=0.25, linewidth=0)
        ax.errorbar(d["load_flows"], d["lat_median"],
                    yerr=[d["lat_median"] - d["lat_ci95_lo"], d["lat_ci95_hi"] - d["lat_median"]],
                    fmt="none", ecolor=COLORS[m], capsize=3, elinewidth=1)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    _plain_log_axis(ax)
    ax.set_xticks(sorted(summary["load_flows"].unique()))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Load: flows per inference call (flows)")
    ax.set_ylabel("Median inference latency (ms)")
    ax.set_title("Inference latency vs load (median, 95 % bootstrap CI)")
    ax.legend(frameon=True, loc="upper left")
    fig.tight_layout()
    return fig


def plot_distribution_boxplot(eval_df: pd.DataFrame, load: int = 4000):
    """Fig 2: latency distribution per model at one load level + 95 % CI of the median."""
    set_style()
    d = eval_df[eval_df["load_flows"] == load]
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    data = [d.loc[d["model"] == m, "latency_ms"].to_numpy() for m in MODELS]
    bp = ax.boxplot(data, positions=range(len(MODELS)), widths=0.55, patch_artist=True, showfliers=True,
                    medianprops=dict(color="black", linewidth=1.6),
                    flierprops=dict(marker="o", markersize=3, alpha=0.5))
    for patch, m in zip(bp["boxes"], MODELS):
        patch.set_facecolor(COLORS[m]); patch.set_alpha(0.65); patch.set_hatch(STYLE[m]["hatch"])
        patch.set_edgecolor("black")
    for i, (m, x) in enumerate(zip(MODELS, data)):
        lo, hi = bootstrap_ci(x, np.median, seed=42 + i)
        med = float(np.median(x))
        ax.errorbar(i + 0.42, med, yerr=[[med - lo], [hi - med]], fmt=STYLE[m]["marker"], color=COLORS[m],
                    markeredgecolor="black", markersize=5, capsize=4, elinewidth=2.2, zorder=5,
                    label="median [95 % bootstrap CI]" if i == 0 else None)
        # CI is narrower than a marker on a 2-decade log axis -> print the numbers as well
        ax.annotate(f"{med:.3g} ms\n[{lo:.3g}; {hi:.3g}]", (i + 0.42, med), xytext=(0, 14 if m != "flat_lr" else 16),
                    textcoords="offset points", ha="center", fontsize=8)
    ax.set_yscale("log")
    _plain_log_axis(ax)
    ax.set_xlim(-0.6, len(MODELS) - 0.3)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([LABELS[m].replace(" (", "\n(") for m in MODELS])
    ax.set_xlabel(f"Model (load = {load} flows per call)")
    ax.set_ylabel("Inference latency (ms)")
    ax.set_title("Latency distribution per model (log scale)")
    ax.legend(loc="center right", frameon=True, fontsize=8.5)
    fig.tight_layout()
    return fig


def main(raw="data/raw/test_run.jsonl", out_dir="reports/figures"):
    matplotlib.use("Agg")  # script mode: no display needed
    eval_df = filter_warmup(load_jsonl(raw))
    summary = aggregate(eval_df)
    for name, fig in (("fig1_latency_vs_load", plot_latency_vs_load(summary)),
                      ("fig2_distribution_boxplot", plot_distribution_boxplot(eval_df))):
        print("saved", save_figure(fig, name, out_dir))
        plt.close(fig)


if __name__ == "__main__":
    main()
