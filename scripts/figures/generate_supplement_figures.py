#!/usr/bin/env python3
"""
Supplement figures from Worker B's data.
  - fig_ablation.pdf: Euclidean vs Hyperbolic × hidden dims
  - fig_baselines.pdf: Simple baselines vs DeepSafe-v3
  - fig_trivial_baselines.pdf: Trivial baselines (all-safe, etc.)

Same NeurIPS-clean style as generate_all_figures.py.
"""
import json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

PROJ = Path(__file__).resolve().parents[2]
OUT_DIR = PROJ / "paper" / "figures"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.8,
    "lines.markersize": 7,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COLOR_EUCLIDEAN = "#D55E00"    # vermillion / warm
COLOR_HYPERBOLIC = "#0072B2"   # blue
COLOR_DEEPSAFE = "#009E73"     # green


# ═══════════════════════════════════════════════════════════════════
# Figure A: Ablation — Euclidean vs Hyperbolic × hidden_dim
# ═══════════════════════════════════════════════════════════════════
def fig_ablation():
    print("Generating Ablation Figure...")
    df = pd.read_csv(PROJ / "reports" / "ablation_table.csv")

    fig, axes = plt.subplots(2, 2, figsize=(9, 7.6))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.92, bottom=0.08, hspace=0.35, wspace=0.25)

    methods = ["Euclidean\n+ BCE", "Hyperbolic\n+ BCE", "Hyperbolic\n+ DeepSafe"]
    colors_bar = [COLOR_EUCLIDEAN, COLOR_HYPERBOLIC, COLOR_DEEPSAFE]
    # All values on same 20K i.i.d. test split (ablation dim=256; DSv3 from seed=42 classifier)
    iid_accs = [0.8003, 0.8111, 0.8798]
    iid_f1s  = [0.8295, 0.8201, 0.8789]

    # ── Accuracy: 3-way bar chart (i.i.d. test) ──
    ax = axes[0, 0]
    bars = ax.bar(methods, iid_accs, color=colors_bar, width=0.5, edgecolor="white", linewidth=0.5)
    for bar, acc in zip(bars, iid_accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{acc:.4f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.set_title("I.I.D. Test Accuracy", fontsize=10, fontweight="bold", pad=12)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    # ── F1 Macro: 3-way bar chart (i.i.d. test) ──
    ax = axes[0, 1]
    bars = ax.bar(methods, iid_f1s, color=colors_bar, width=0.5, edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, iid_f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("F1 Macro", fontsize=10)
    ax.set_title("F1 Macro Comparison", fontsize=10, fontweight="bold", pad=12)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    # ── ROC Curves: Euc vs Hyp at hidden_dim=256 ──
    ax = axes[1, 0]
    roc_data = np.load(PROJ / "reports" / "roc_curve_data.npz")
    euc_fpr, euc_tpr = roc_data["euc_fpr"], roc_data["euc_tpr"]
    hyp_fpr, hyp_tpr = roc_data["hyp_fpr"], roc_data["hyp_tpr"]
    euc_auc, hyp_auc = float(roc_data["euc_auc"]), float(roc_data["hyp_auc"])

    ax.plot(euc_fpr, euc_tpr, color=COLOR_EUCLIDEAN, linewidth=1.5,
            label=f"Euclidean + BCE (AUC={euc_auc:.4f})")
    ax.plot(hyp_fpr, hyp_tpr, color=COLOR_HYPERBOLIC, linewidth=1.5,
            linestyle="--", label=f"Hyperbolic + BCE (AUC={hyp_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linewidth=0.6, linestyle=":",
            alpha=0.6, label="Random (AUC=0.5)")

    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title("ROC Curve (hidden_dim=256, i.i.d. test set)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    # ── Inference Speed ──
    ax = axes[1, 1]
    for typ, color, marker, label in [
        ("euclidean", COLOR_EUCLIDEAN, "s", "Euclidean"),
        ("hyperbolic", COLOR_HYPERBOLIC, "D", "Hyperbolic"),
    ]:
        sub = df[df["type"] == typ]
        ax.plot(sub["hidden_dim"], sub["time_s"], marker=marker,
                color=color, linewidth=2, markersize=8, label=label)
    ax.set_xlabel("Hidden Dimension", fontsize=10)
    ax.set_ylabel("Inference Speed(samples/s)", fontsize=10)
    ax.set_title("Inference Speed", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xticks(df["hidden_dim"].unique())
    ax.grid(True, alpha=0.3)

    fig.suptitle("Ablation: Euclidean vs Hyperbolic Projection",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.savefig(OUT_DIR / "fig_ablation.pdf", facecolor="white")
    fig.savefig(OUT_DIR / "fig_ablation.png", facecolor="white", dpi=300)
    plt.close(fig)
    print("  Saved fig_ablation.pdf")


# ═══════════════════════════════════════════════════════════════════
# Figure B: Baselines comparison — LR, MLP, EuclideanProj vs DeepSafe
# ═══════════════════════════════════════════════════════════════════
def fig_baselines():
    print("Generating Baselines Comparison Figure...")
    df = pd.read_csv(PROJ / "reports" / "baselines_table.csv")

    # Load DeepSafe data for comparison
    with open(PROJ / "valuation" / "deepsafe_v3_results.json") as f:
        ds_data = json.load(f)

    benchmarks = df["benchmark"].unique()
    bench_labels = [b.replace("100PoisonMpts", "100Poison\nMpts")
                     .replace("AegisAI-", "AegisAI\n")
                     .replace("BeaverTails", "Beaver\nTails")
                     .replace("DoNotAnswer", "DoNot\nAnswer")
                     .replace("WildGuardMix", "WildGuard\nMix")
                    for b in benchmarks]

    models = df["model"].unique()
    n_benches = len(benchmarks)
    n_models = len(models) + 1  # +1 for DeepSafe
    width = 0.8 / n_models
    x = np.arange(n_benches)

    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)

    colors_baseline = {
        "LR-4096d": "#E69F00",
        "MLP-4096-512-2": "#56B4E9",
        "EuclideanProj-NN": "#CC79A7",
    }

    for i, model in enumerate(models):
        accs = []
        for bench in benchmarks:
            row = df[(df["benchmark"] == bench) & (df["model"] == model)]
            accs.append(row["accuracy"].values[0] if len(row) > 0 else 0)
        ax.bar(x + i * width, accs, width,
               label=model, color=colors_baseline.get(model, "#999"),
               alpha=0.8, zorder=2)

    # DeepSafe-v3
    ds_accs = []
    for bench in benchmarks:
        bench_key = bench
        if bench in ds_data and "DeepSafe-v3" in ds_data[bench]:
            ds_accs.append(ds_data[bench]["DeepSafe-v3"]["accuracy"])
        else:
            ds_accs.append(0)
    i_ds = len(models)
    ax.bar(x + i_ds * width, ds_accs, width,
           label="DeepSafe-v3", color=COLOR_DEEPSAFE, alpha=1.0,
           edgecolor="black", linewidth=1.2, zorder=3)

    ax.set_xticks(x + width * n_models / 2)
    ax.set_xticklabels(bench_labels, fontsize=8)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.legend(ncol=2, fontsize=8, loc="upper right", frameon=True, framealpha=0.9)

    fig.savefig(OUT_DIR / "fig_baselines.pdf", facecolor="white")
    fig.savefig(OUT_DIR / "fig_baselines.png", facecolor="white", dpi=300)
    plt.close(fig)
    print("  Saved fig_baselines.pdf")


# ═══════════════════════════════════════════════════════════════════
# Figure C: Trivial baselines — sanity check
# ═══════════════════════════════════════════════════════════════════
def fig_trivial_baselines():
    print("Generating Trivial Baselines Figure...")
    df = pd.read_csv(PROJ / "reports" / "trivial_baselines.csv")

    # Load DeepSafe data
    with open(PROJ / "valuation" / "deepsafe_v3_results.json") as f:
        ds_data = json.load(f)

    benchmarks = df["benchmark"].unique()
    bench_labels = [b.replace("100PoisonMpts", "100Poison\nMpts")
                     .replace("AegisAI-", "AegisAI\n")
                     .replace("BeaverTails", "Beaver\nTails")
                     .replace("DoNotAnswer", "DoNot\nAnswer")
                     .replace("WildGuardMix", "WildGuard\nMix")
                    for b in benchmarks]

    trivial_models = ["predict_all_safe", "majority_class"]

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

    n_benches = len(benchmarks)
    n_items = len(trivial_models) + 1  # + DeepSafe
    width = 0.8 / n_items
    x = np.arange(n_benches)

    colors_trivial = {"predict_all_safe": "#CCCCCC", "majority_class": "#AAAAAA"}

    for i, model in enumerate(trivial_models):
        accs = []
        for bench in benchmarks:
            row = df[(df["benchmark"] == bench) & (df["model"] == model)]
            accs.append(row["accuracy"].values[0] if len(row) > 0 else 0)
        # Use hatched bars for trivial baselines
        ax.bar(x + i * width, accs, width,
               label=model.replace("_", " "),
               color=colors_trivial[model],
               hatch="///" if model == "predict_all_safe" else "\\\\\\",
               edgecolor="#666666", linewidth=0.5, alpha=0.7, zorder=2)

    # DeepSafe-v3
    ds_accs = []
    for bench in benchmarks:
        bench_key = bench
        if bench in ds_data and "DeepSafe-v3" in ds_data[bench]:
            ds_accs.append(ds_data[bench]["DeepSafe-v3"]["accuracy"])
        else:
            ds_accs.append(0)
    ax.bar(x + len(trivial_models) * width, ds_accs, width,
           label="DeepSafe-v3", color=COLOR_DEEPSAFE, alpha=1.0,
           edgecolor="black", linewidth=1.2, zorder=3)

    ax.set_xticks(x + width * n_items / 2)
    ax.set_xticklabels(bench_labels, fontsize=8)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.legend(fontsize=9, loc="upper right", frameon=True, framealpha=0.9)

    fig.savefig(OUT_DIR / "fig_trivial_baselines.pdf", facecolor="white")
    fig.savefig(OUT_DIR / "fig_trivial_baselines.png", facecolor="white", dpi=300)
    plt.close(fig)
    print("  Saved fig_trivial_baselines.pdf")


# ═══════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("Generating Supplement Figures from Worker B Data")
    print("=" * 60)
    fig_ablation()
    fig_baselines()
    fig_trivial_baselines()
    print(f"\nAll supplement figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
