"""Generate publication-quality figures for SafeCL NeurIPS 2026 paper.

Uses scientific-visualization skill patterns: colorblind-safe palette,
proper figure sizing, vector export, multi-panel layouts.

Output: paper/figures/*.pdf
"""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.seed import set_seed

set_seed(42)

# ---- Colorblind-safe palette (Okabe-Ito) ----
C_SAFE = "#009E73"       # green
C_BENIGN = "#E69F00"     # orange-yellow
C_MALICIOUS = "#D55E00"  # vermillion
C_SAFECL = "#CC79A7"     # reddish-purple (for SafeCL bars)
C_BASELINE = "#56B4E9"   # sky blue (for baselines)
C_LINE = "#0072B2"       # blue
C_GRAY = "#999999"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

EMBED_DIR = os.path.join(os.path.dirname(__file__), "..", "embeddings/qwen3-embedding-0.6B")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "..", "reports/comprehensive/evaluation_report.json")
COUNTERFACTUAL_PATH = os.path.join(os.path.dirname(__file__), "..", "reports/comprehensive/counterfactuals.csv")

# ---- Load data ----
X_test = np.load(os.path.join(EMBED_DIR, "test.npy"))
y_test = np.load(os.path.join(EMBED_DIR, "test_labels.npy"))

# Intent labels
try:
    from src.safecl.intent import IntentAnnotator
    df_test = pd.read_parquet("data/processed/test.parquet")
    annotator = IntentAnnotator()
    df_test = annotator.annotate(df_test)
    y_intent = df_test["intent"].values
except Exception:
    y_intent = y_test

# ---- Load SafeCL V9 projected embeddings ----
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.safecl import ProjectionHeadManager
import torch

proj_head = ProjectionHeadManager.load("models/safecl_v9/qwen3-embedding-0.6B/projection_head.pkl", device="cuda")
X_proj = ProjectionHeadManager.project(proj_head, X_test, device="cuda")

# ---- Load evaluation report ----
with open(REPORT_PATH) as f:
    report = json.load(f)

# ---- Load counterfactuals ----
df_cf = pd.read_csv(COUNTERFACTUAL_PATH) if os.path.exists(COUNTERFACTUAL_PATH) else None


# =====================================================================
# Figure 1: t-SNE comparison (2 panels side-by-side)
# =====================================================================
def fig1_tsne():
    print("Generating Figure 1: t-SNE comparison...")
    n = 2500
    idx = np.random.RandomState(42).choice(len(X_test), n, replace=False)
    X_orig_sub = X_test[idx]
    X_proj_sub = X_proj[idx]
    y_sub = y_intent[idx]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))

    for ax_idx, (X, title) in enumerate([
        (X_orig_sub, "Original Qwen3-Embedding-0.6B"),
        (X_proj_sub, "SafeCL Projected (Intent-Aware)"),
    ]):
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000, n_jobs=1)
        X_2d = tsne.fit_transform(StandardScaler().fit_transform(X))

        colors = [C_SAFE, C_BENIGN, C_MALICIOUS]
        labels = ["Safe", "Pseudo-Harmful", "Malicious"]
        markers = ["o", "s", "D"]
        for i, c, m, lab in zip([0, 1, 2], colors, markers, labels):
            mask = y_sub == i
            if mask.sum() > 0:
                axes[ax_idx].scatter(X_2d[mask, 0], X_2d[mask, 1],
                                     c=c, marker=m, label=lab, alpha=0.5, s=3,
                                     edgecolors="none", rasterized=True)

        axes[ax_idx].set_title(title, fontsize=10, fontweight="bold", pad=6)
        axes[ax_idx].legend(markerscale=5, fontsize=7, loc="upper right", framealpha=0.9,
                           handletextpad=0.5, borderpad=0.3)
        axes[ax_idx].set_xticks([])
        axes[ax_idx].set_yticks([])
        # Remove spines for cleaner look
        for spine in axes[ax_idx].spines.values():
            spine.set_visible(False)

    # Add panel labels
    for i, ax in enumerate(axes):
        ax.text(-0.02, 1.04, chr(65 + i), transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="bottom")

    fig.suptitle("Intent-Driven Safety Representation Learning",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_tsne.pdf"), format="pdf", dpi=300)
    fig.savefig(os.path.join(OUT, "fig1_tsne.png"), format="png", dpi=300)
    plt.close()
    print("  -> fig1_tsne.pdf")


# =====================================================================
# Figure 2: Benchmark comparison bar chart
# =====================================================================
def fig2_benchmark():
    print("Generating Figure 2: Benchmark comparison...")
    baselines = report["baseline_classifiers"]
    safecl = report["safecl_classifier"]

    metrics_keys = ["accuracy", "f1_macro", "roc_auc"]
    metrics_labels = ["Accuracy", "F1 Score", "ROC AUC"]
    n_metrics = len(metrics_keys)

    x = np.arange(n_metrics)
    width = 0.15

    fig, ax = plt.subplots(figsize=(7.0, 3.5))

    # Baselines
    baseline_names = list(baselines.keys())
    baseline_colors = ["#B0BEC5", "#90A4AE", "#78909C", "#607D8B"]

    for i, (name, color) in enumerate(zip(baseline_names, baseline_colors)):
        values = [baselines[name].get(k, 0) for k in metrics_keys]
        bars = ax.bar(x + i * width, values, width, label=name, color=color, alpha=0.85,
                      edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=6, rotation=90)

    # SafeCL (highlighted)
    safecl_values = [safecl.get(k, 0) for k in metrics_keys]
    offset = len(baselines)
    bars = ax.bar(x + offset * width, safecl_values, width,
                  label="SafeCL (Ours)", color=C_SAFECL, alpha=0.95,
                  edgecolor="#8B0060", linewidth=1.5)
    for bar, val in zip(bars, safecl_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7, rotation=90,
                fontweight="bold", color="#8B0060")

    ax.set_ylabel("Score", fontsize=10)
    ax.set_xticks(x + width * (len(baselines) + 1 - 1) / 2)
    ax.set_xticklabels(metrics_labels, fontsize=10)
    ax.legend(loc="lower right", fontsize=7.5, ncol=3, framealpha=0.9)
    ax.set_ylim(0.70, 0.98)
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_title("Classification Performance Comparison", fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_benchmark.pdf"), format="pdf", dpi=300)
    fig.savefig(os.path.join(OUT, "fig2_benchmark.png"), format="png", dpi=300)
    plt.close()
    print("  -> fig2_benchmark.pdf")


# =====================================================================
# Figure 3: Counterfactual separation + Per-intent accuracy (2 panels)
# =====================================================================
def fig3_intent_analysis():
    print("Generating Figure 3: Intent analysis...")

    fig = plt.figure(figsize=(7.0, 5.5))

    # Panel A: Counterfactual pairs
    ax1 = fig.add_subplot(2, 1, 1)
    if df_cf is not None:
        topics = [t[:30] for t in df_cf["topic"].tolist()]
        ratios = df_cf["euc_ratio"].tolist()
        orig_euc = df_cf["orig_euclidean"].tolist()
        proj_euc = df_cf["proj_euclidean"].tolist()

        colors = [C_SAFE if r > 1.0 else "#E0E0E0" if r > 0.9 else C_MALICIOUS for r in ratios]
        ax1.barh(range(len(topics)), ratios, color=colors, alpha=0.85, height=0.7,
                 edgecolor="white", linewidth=0.3)
        ax1.axvline(x=1.0, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
        ax1.set_yticks(range(len(topics)))
        ax1.set_yticklabels(topics, fontsize=6.5)
        ax1.set_xlabel("Euclidean Distance Ratio (projected / original)", fontsize=9)
        ax1.set_title("Counterfactual Pair Separation", fontsize=10, fontweight="bold")
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        # Add ratio annotation
        for i, (r, oe, pe) in enumerate(zip(ratios, orig_euc, proj_euc)):
            color = C_SAFE if r > 1.0 else C_MALICIOUS
            ax1.text(r + 0.03, i, f"{r:.2f}x", va="center", fontsize=6, color=color, fontweight="bold")

        improved = sum(1 for r in ratios if r > 1.0)
        ax1.text(0.98, 0.02, f"{improved}/{len(ratios)} pairs improved",
                 transform=ax1.transAxes, fontsize=7, ha="right", va="bottom",
                 fontstyle="italic")

    # Panel B: Intent accuracy breakdown
    ax2 = fig.add_subplot(2, 1, 2)
    intent_data = report.get("per_intent_accuracy", [])
    if intent_data:
        names = [d["intent"].replace("_", "-").title() for d in intent_data]
        accs = [d["accuracy"] for d in intent_data]
        counts = [d["count"] for d in intent_data]
        colors = [C_SAFE, C_BENIGN, C_MALICIOUS]

        bars = ax2.bar(names, accs, color=colors, alpha=0.85, width=0.5,
                       edgecolor="black", linewidth=0.5)
        for bar, acc, count in zip(bars, accs, counts):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{acc:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.05,
                     f"n={count:,}", ha="center", va="top", fontsize=8, color="white")

        ax2.set_ylabel("Accuracy", fontsize=9)
        ax2.set_title("Per-Intent Classification Accuracy", fontsize=10, fontweight="bold")
        ax2.set_ylim(0, 1.0)
        ax2.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

    # Panel labels
    ax1.text(-0.08, 1.05, "A", transform=ax1.transAxes,
             fontsize=12, fontweight="bold", va="bottom")
    ax2.text(-0.08, 1.05, "B", transform=ax2.transAxes,
             fontsize=12, fontweight="bold", va="bottom")

    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_intent.pdf"), format="pdf", dpi=300)
    fig.savefig(os.path.join(OUT, "fig3_intent.png"), format="png", dpi=300)
    plt.close()
    print("  -> fig3_intent.pdf")


# =====================================================================
# Figure 4: Representation quality before/after
# =====================================================================
def fig4_representation():
    print("Generating Figure 4: Representation quality...")
    rep = report["representation_quality"]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

    metrics = ["silhouette", "separation_ratio"]
    titles = ["Silhouette Score", "Separation Ratio"]
    y_labels = ["Score", "Ratio (inter / intra)"]

    for ax_idx, (metric, title, ylab) in enumerate(zip(metrics, titles, y_labels)):
        orig_val = rep["original"][metric]
        proj_val = rep["projected"][metric]
        gain = rep["improvement"]["silhouette_gain" if metric == "silhouette" else "separation_gain"]

        bars = axes[ax_idx].bar(["Original", "SafeCL\nProjected"], [orig_val, proj_val],
                                color=[C_GRAY, C_SAFECL], alpha=0.85, width=0.45,
                                edgecolor="black", linewidth=0.5)

        for bar, val in zip(bars, [orig_val, proj_val]):
            axes[ax_idx].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                             f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

        # Gain annotation
        axes[ax_idx].annotate(f"{gain:.0f}x improvement",
                             xy=(1, proj_val), xytext=(1.3, proj_val + max(orig_val, proj_val) * 0.5),
                             fontsize=8, ha="center", fontweight="bold", color=C_SAFECL,
                             arrowprops=dict(arrowstyle="->", color=C_SAFECL, lw=1.5))

        axes[ax_idx].set_title(title, fontsize=10, fontweight="bold")
        axes[ax_idx].set_ylabel(ylab, fontsize=9)
        axes[ax_idx].grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
        axes[ax_idx].spines["top"].set_visible(False)
        axes[ax_idx].spines["right"].set_visible(False)

    for i, ax in enumerate(axes):
        ax.text(-0.05, 1.05, chr(65 + i), transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="bottom")

    fig.suptitle("Representation Quality Improvement", fontsize=12, fontweight="bold", y=1.05)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_representation.pdf"), format="pdf", dpi=300)
    fig.savefig(os.path.join(OUT, "fig4_representation.png"), format="png", dpi=300)
    plt.close()
    print("  -> fig4_representation.pdf")


if __name__ == "__main__":
    fig1_tsne()
    fig2_benchmark()
    fig3_intent_analysis()
    fig4_representation()
    print("\nAll figures saved to paper/figures/")
