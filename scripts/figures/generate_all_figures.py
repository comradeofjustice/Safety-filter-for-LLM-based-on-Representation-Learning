#!/usr/bin/env python3
"""
Generate all publication-quality figures for DeepSafe v3 paper (NeurIPS 2026).
Worker C — Figures 1-5.

Output: paper/figures/{fig_benchmark_accuracy,fig_radar,fig_speed_vs_accuracy,fig_head_to_head,fig_tsne}.pdf

Design principles:
  - NeurIPS-friendly: clean, minimal, high contrast
  - Colorblind-aware palette (Okabe-Ito / Wong)
  - Consistent font sizes, line weights, and styling
  - All outputs as vector PDF + companion PNG
"""
import json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

PROJ = Path(__file__).resolve().parents[2]
OUT_DIR = PROJ / "paper" / "figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
# Global style — NeurIPS clean
# ═══════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Palatino"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "figure.titlesize": 13,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# 10-model color palette (Okabe-Ito derived, colorblind-friendly)
COLORS = {
    "DeepSafe-v3":             "#0072B2",  # blue (ours, bold)
    "Qwen3Guard-Gen-8B":       "#E69F00",  # orange
    "Qwen3Guard-Gen-4B":       "#F5C710",  # gold/yellow
    "Llama-Guard-3-8B":        "#009E73",  # green
    "Llama-Guard-3-1B":        "#56B4E9",  # sky blue
    "WildGuard":               "#D55E00",  # vermillion
    "ShieldGemma-9B":          "#CC79A7",  # reddish purple
    "ShieldGemma-2B":          "#F0A0C0",  # pink
    "Granite-Guardian-3.1-8B": "#8C564B",  # brown
    "Beaver-Dam-7B":           "#E57373",  # red
}

BENCHMARK_ORDER = [
    "WildGuardMix", "AegisAI-v2", "AegisAI-v1", "ToxicChat",
    "XSTest", "BeaverTails", "XGuard", "DoNotAnswer", "100PoisonMpts",
]

MODEL_DISPLAY = {
    "DeepSafe-v3": "DeepSafe-v3",
    "Qwen3Guard-Gen-8B": "Qwen3Guard-8B",
    "Qwen3Guard-Gen-4B": "Qwen3Guard-4B",
    "Llama-Guard-3-8B": "Llama-Guard-3-8B",
    "Llama-Guard-3-1B": "Llama-Guard-3-1B",
    "WildGuard": "WildGuard",
    "ShieldGemma-9B": "ShieldGemma-9B",
    "ShieldGemma-2B": "ShieldGemma-2B",
    "Granite-Guardian-3.1-8B": "Granite-G-3.1",
    "Beaver-Dam-7B": "Beaver-Dam-7B",
}

SHORT_BENCH = {
    "WildGuardMix": "WildGuard\nMix",
    "AegisAI-v2": "AegisAI\nv2",
    "AegisAI-v1": "AegisAI\nv1",
    "ToxicChat": "ToxicChat",
    "XSTest": "XSTest",
    "BeaverTails": "Beaver\nTails",
    "XGuard": "XGuard",
    "DoNotAnswer": "DoNot\nAnswer",
    "100PoisonMpts": "100Poison\nMpts",
}

SOTA_MODELS = [
    "Qwen3Guard-Gen-4B", "Qwen3Guard-Gen-8B",
    "Llama-Guard-3-8B", "Llama-Guard-3-1B",
    "ShieldGemma-9B", "ShieldGemma-2B",
    "WildGuard", "Granite-Guardian-3.1-8B", "Beaver-Dam-7B",
]

BASELINE_MODELS = [
    "SafeCL-v9", "Embedding+logistic-8B", "Embedding+mlp-8B",
]


def load_data():
    """Load all evaluation data."""
    with open(PROJ / "valuation" / "deepsafe_v3_results.json") as f:
        ds_results = json.load(f)
    with open(PROJ / "valuation" / "all_results.json") as f:
        all_results = json.load(f)

    # Merge: prefer deepsafe results, fall back to all_results
    merged = {}
    for bench in ds_results:
        if bench not in all_results:
            merged[bench] = dict(ds_results[bench])
        else:
            merged[bench] = {}
            for model in set(list(ds_results[bench].keys()) + list(all_results[bench].keys())):
                if model in ds_results[bench]:
                    merged[bench][model] = ds_results[bench][model]
                elif bench in all_results and model in all_results[bench]:
                    merged[bench][model] = all_results[bench][model]

    return merged


def get_model_color(model_name):
    """Get consistent color for a model. Exact match first, then fallback."""
    if model_name in COLORS:
        return COLORS[model_name]
    if "DeepSafe" in model_name:
        return COLORS["DeepSafe-v3"]
    return "#999999"


# ═══════════════════════════════════════════════════════════════════
# Figure 1 — Bar chart: accuracy per benchmark × model (with error bars)
# ═══════════════════════════════════════════════════════════════════
def fig1_bar_chart(data):
    print("Generating Figure 1: Benchmark Accuracy Bar Chart...")
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)

    benchmarks_plot = [b for b in BENCHMARK_ORDER if b in data]
    # All 10 models: DeepSafe + 9 SOTA
    models_plot = ["DeepSafe-v3"] + SOTA_MODELS

    n_benches = len(benchmarks_plot)
    n_models = len(models_plot)
    width = 0.85 / n_models
    x = np.arange(n_benches)

    print(f"  Models: {n_models}, Benchmarks: {n_benches}")

    # Load CI data for error bars
    ci_df = None
    ci_path = PROJ / "reports" / "stats_with_ci.csv"
    if ci_path.exists():
        ci_df = pd.read_csv(ci_path)

    for i, model in enumerate(models_plot):
        accs = []
        errs = []
        for bench in benchmarks_plot:
            if bench in data and model in data[bench]:
                accs.append(data[bench][model].get("accuracy", 0))
            else:
                accs.append(0)

            # Get CI/error from stats
            err = 0
            if ci_df is not None:
                if model == "DeepSafe-v3":
                    row = ci_df[(ci_df["benchmark"] == bench) & (ci_df["model"] == model)]
                else:
                    row = ci_df[(ci_df["benchmark"] == bench) & (ci_df["model"] == model)]
                if len(row) > 0 and row["accuracy_std"].iloc[0] > 0:
                    err = row["accuracy_std"].iloc[0]
            errs.append(err)

        color = get_model_color(model)
        label = MODEL_DISPLAY.get(model, model)
        alpha = 1.0 if model == "DeepSafe-v3" else 0.75
        edge = "black" if model == "DeepSafe-v3" else "none"
        lw = 1.2 if model == "DeepSafe-v3" else 0

        bars = ax.bar(x + i * width, accs, width, label=label,
                      color=color, alpha=alpha, edgecolor=edge, linewidth=lw, zorder=2)

        if model == "DeepSafe-v3":
            ax.errorbar(x + i * width, accs, yerr=errs, fmt="none",
                        ecolor="black", capsize=3, capthick=1, linewidth=0.8, zorder=3)

    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels([SHORT_BENCH.get(b, b) for b in benchmarks_plot], fontsize=7.5)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.grid(axis="y", alpha=0.3, zorder=0)

    # Highlight DeepSafe-v3 in legend
    handles, labels = ax.get_legend_handles_labels()
    # Reorder: DeepSafe-v3 first
    ds_idx = labels.index("DeepSafe-v3") if "DeepSafe-v3" in labels else 0
    order = [ds_idx] + [j for j in range(len(handles)) if j != ds_idx]
    ax.legend([handles[j] for j in order], [labels[j] for j in order],
              ncol=3, loc="upper right", frameon=True, framealpha=0.9,
              fontsize=6, columnspacing=0.8, handlelength=1.2, handletextpad=0.5)

    fig.savefig(OUT_DIR / "fig_benchmark_accuracy.pdf", facecolor="white")
    fig.savefig(OUT_DIR / "fig_benchmark_accuracy.png", facecolor="white", dpi=300)
    plt.close(fig)
    print("  Saved fig_benchmark_accuracy.pdf")


# ═══════════════════════════════════════════════════════════════════
# Figure 2 — Radar chart with CI
# ═══════════════════════════════════════════════════════════════════
def fig2_radar(data):
    print("Generating Figure 2: Radar Chart...")
    ci_df = pd.read_csv(PROJ / "reports" / "stats_with_ci.csv")

    benchmarks_radar = [b for b in BENCHMARK_ORDER if b in data]
    n_benches = len(benchmarks_radar)

    # Compute angles
    angles = np.linspace(0, 2 * np.pi, n_benches, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(7.5, 7), subplot_kw={"projection": "polar"},
                           constrained_layout=True)

    # Models for radar (fewer for readability)
    radar_models = ["DeepSafe-v3", "Qwen3Guard-Gen-8B", "Llama-Guard-3-8B",
                    "WildGuard", "ShieldGemma-9B", "Granite-Guardian-3.1-8B"]

    for model in radar_models:
        accs = []
        ci_lows = []
        ci_highs = []
        for bench in benchmarks_radar:
            if model == "DeepSafe-v3":
                row = ci_df[(ci_df["benchmark"] == bench) & (ci_df["model"] == model)]
            elif bench in data and model in data[bench]:
                row = ci_df[(ci_df["benchmark"] == bench) & (ci_df["model"] == model)]
            else:
                row = pd.DataFrame()

            if len(row) > 0:
                accs.append(row["accuracy_mean"].iloc[0])
                ci_lows.append(row["accuracy_ci_low"].iloc[0] if pd.notna(row["accuracy_ci_low"].iloc[0]) else row["accuracy_mean"].iloc[0])
                ci_highs.append(row["accuracy_ci_high"].iloc[0] if pd.notna(row["accuracy_ci_high"].iloc[0]) else row["accuracy_mean"].iloc[0])
            else:
                accs.append(0)
                ci_lows.append(0)
                ci_highs.append(0)

        accs += accs[:1]  # close
        ci_lows += ci_lows[:1]
        ci_highs += ci_highs[:1]

        color = get_model_color(model)
        label = MODEL_DISPLAY.get(model, model)
        lw = 2.5 if model == "DeepSafe-v3" else 1.5
        ls = "-" if model == "DeepSafe-v3" else "--"
        alpha = 1.0 if model == "DeepSafe-v3" else 0.7

        ax.plot(angles, accs, "o-", color=color, label=label, linewidth=lw,
                linestyle=ls, markersize=4 if model != "DeepSafe-v3" else 7, alpha=alpha)

        if model == "DeepSafe-v3":
            ax.fill_between(angles, ci_lows, ci_highs, alpha=0.12, color=color, zorder=0)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([SHORT_BENCH.get(b, b).replace("\n", " ") for b in benchmarks_radar],
                       fontsize=7.5)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.08), fontsize=7, frameon=True,
              framealpha=0.9, ncol=1)

    fig.savefig(OUT_DIR / "fig_radar.pdf", facecolor="white")
    fig.savefig(OUT_DIR / "fig_radar.png", facecolor="white", dpi=300)
    plt.close(fig)
    print("  Saved fig_radar.pdf")


# ═══════════════════════════════════════════════════════════════════
# Figure 3 — Speed vs Accuracy scatter (with error bars for DeepSafe)
# ═══════════════════════════════════════════════════════════════════
def fig3_speed_vs_accuracy(data):
    print("Generating Figure 3: Speed vs Accuracy...")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.94, bottom=0.13)

    ci_df = pd.read_csv(PROJ / "reports" / "stats_with_ci.csv")

    # Only DeepSafe + 9 SOTA models
    PLOT_MODELS = set(["DeepSafe-v3"] + SOTA_MODELS)
    DEEPSAFE_THROUGHPUT = 55.0  # manually measured samples/s

    # Aggregate per model: mean accuracy and throughput across all benchmarks
    for model_name in sorted(set(
        [m for b in data for m in data[b] if m in PLOT_MODELS]
    )):
        weighted_accs = []
        total_samples = 0
        total_time = 0

        is_ds = "DeepSafe" in model_name

        for bench in data:
            if bench in data and model_name in data[bench]:
                m = data[bench][model_name]
                if "accuracy" in m:
                    n_samples = m.get("samples", 0)
                    if n_samples > 0:
                        weighted_accs.append(m["accuracy"] * n_samples)
                        total_samples += n_samples
                    if not is_ds and "inference_time_s" in m:
                        total_time += m["inference_time_s"]

        if total_samples < 500:
            continue
        if not is_ds and total_time == 0:
            continue

        mean_acc = sum(weighted_accs) / total_samples
        throughput = DEEPSAFE_THROUGHPUT if is_ds else (total_samples / total_time)

        # Get CI for DeepSafe
        acc_std = 0
        if is_ds and ci_df is not None:
            ds_ci = ci_df[ci_df["model"] == "DeepSafe-v3"]
            if len(ds_ci) > 0:
                acc_std = ds_ci["accuracy_std"].mean()

        color = get_model_color(model_name)
        label = MODEL_DISPLAY.get(model_name, model_name)
        size = 100 if is_ds else 60
        marker = "D" if is_ds else "o"
        zorder = 3 if is_ds else 2
        edge = "black" if is_ds else "none"
        lw = 1.5 if is_ds else 0

        ax.errorbar(throughput, mean_acc, yerr=acc_std if is_ds else 0,
                    fmt=marker, color=color, markersize=np.sqrt(size / np.pi) * 2,
                    markeredgecolor=edge, markeredgewidth=lw, capsize=4 if is_ds else 0,
                    capthick=1, label=label, zorder=zorder, alpha=0.9)

    ax.set_xlabel("Throughput (samples/s, higher is better)", fontsize=11)
    ax.set_ylabel("Weighted Mean Accuracy across Benchmarks", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(0.3, 1.02)
    ax.legend(loc="upper right", fontsize=7, markerscale=1.2, frameon=True,
              framealpha=0.9, ncol=2)

    fig.savefig(OUT_DIR / "fig_speed_vs_accuracy.pdf", facecolor="white", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig_speed_vs_accuracy.png", facecolor="white", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig_speed_vs_accuracy.pdf")


# ═══════════════════════════════════════════════════════════════════
# Figure 4 — Head-to-head matrix
# ═══════════════════════════════════════════════════════════════════
def fig4_head_to_head(data):
    print("Generating Figure 4: Head-to-Head Comparison Matrix...")
    # Build head-to-head: for each model pair, count how many benchmarks A > B

    all_models = ["DeepSafe-v3"] + SOTA_MODELS
    available = [m for m in all_models if any(
        m in data.get(b, {}) for b in data
    )]
    n = len(available)
    matrix = np.zeros((n, n))

    for i, model_a in enumerate(available):
        for j, model_b in enumerate(available):
            if i == j:
                matrix[i, j] = 0
                continue
            wins = 0
            comparisons = 0
            for bench in data:
                if model_a in data[bench] and model_b in data[bench]:
                    acc_a = data[bench][model_a].get("accuracy", 0)
                    acc_b = data[bench][model_b].get("accuracy", 0)
                    if acc_a > acc_b:
                        wins += 1
                    comparisons += 1
            matrix[i, j] = wins / max(comparisons, 1) if comparisons > 0 else 0.5

    fig, ax = plt.subplots(figsize=(9, 7.5), constrained_layout=True)

    cmap = plt.cm.RdBu_r
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    # Labels
    labels = [MODEL_DISPLAY.get(m, m) for m in available]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5)
    ax.set_yticklabels(labels, fontsize=7.5)

    # Highlight DeepSafe row
    for j in range(n):
        ax.add_patch(plt.Rectangle((j - 0.5, 0 - 0.5), 1, 1,
                                    fill=False, edgecolor="black", linewidth=2, zorder=3))

    # Add text annotations
    for i in range(n):
        for j in range(n):
            if i == j:
                text = "-"
                color = "gray"
            else:
                text = f"{matrix[i, j]:.2f}"
                color = "white" if matrix[i, j] < 0.3 or matrix[i, j] > 0.7 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=7.5,
                    color=color, fontweight="bold" if i == j else "normal")

    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Win Rate (row over column)", fontsize=9)
    cbar.ax.tick_params(labelsize=7)

    ax.set_xlabel("Column Model", fontsize=10)
    ax.set_ylabel("Row Model", fontsize=10)

    fig.savefig(OUT_DIR / "fig_head_to_head.pdf", facecolor="white")
    fig.savefig(OUT_DIR / "fig_head_to_head.png", facecolor="white", dpi=300)
    plt.close(fig)
    print("  Saved fig_head_to_head.pdf")


# ═══════════════════════════════════════════════════════════════════
# Figure 5 — t-SNE visualization with silhouette annotation
# ═══════════════════════════════════════════════════════════════════
def fig5_tsne():
    print("Generating Figure 5: t-SNE with silhouette...")
    # Use the DeepSafe augmented embeddings for t-SNE
    emb_path = PROJ / "embeddings" / "deepsafe_v3_augmented"
    test_emb = np.load(emb_path / "test.npy")
    test_labels = np.load(emb_path / "test_labels.npy")

    # Subsample for t-SNE speed
    n_max = 3000
    if len(test_emb) > n_max:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(test_emb), n_max, replace=False)
        test_emb = test_emb[idx]
        test_labels = test_labels[idx]

    print(f"  t-SNE on {len(test_emb)} samples...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=6, verbose=0)
    emb_2d = tsne.fit_transform(test_emb)

    # Compute silhouette score
    unique_labels = np.unique(test_labels)
    if len(unique_labels) >= 2:
        sil = silhouette_score(emb_2d, test_labels)
    else:
        sil = 0.0

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)

    colors_map = {0: "#2E86AB", 1: "#A23B72", 2: "#F18F01"}
    labels_map = {0: "Safe", 1: "Unsafe", 2: "Borderline"}

    for label_val in sorted(unique_labels):
        mask = test_labels == label_val
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                   c=colors_map.get(label_val, "#999999"),
                   label=labels_map.get(label_val, f"Class {label_val}"),
                   alpha=0.5, s=8, edgecolors="none", rasterized=True)

    ax.set_xlabel("t-SNE Component 1", fontsize=11)
    ax.set_ylabel("t-SNE Component 2", fontsize=11)
    ax.legend(loc="upper right", fontsize=9, markerscale=2.5)

    # Silhouette annotation
    ax.text(0.02, 0.98, f"Silhouette Score = {sil:.3f}",
            transform=ax.transAxes, fontsize=10, fontweight="bold",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="gray", alpha=0.85))

    fig.savefig(OUT_DIR / "tsne_comparison.pdf", facecolor="white")
    fig.savefig(OUT_DIR / "tsne_comparison.png", facecolor="white", dpi=300)
    plt.close(fig)
    print(f"  Saved tsne_comparison.pdf (Silhouette = {sil:.3f})")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("Generating All Figures for DeepSafe v3 Paper")
    print("=" * 60)

    data = load_data()
    print(f"Loaded data: {len(data)} benchmarks, "
          f"{len(set(m for b in data for m in data[b]))} models\n")

    fig1_bar_chart(data)
    fig2_radar(data)
    fig3_speed_vs_accuracy(data)
    fig4_head_to_head(data)
    fig5_tsne()

    print(f"\nAll figures saved to {OUT_DIR}/")
    for f in sorted(OUT_DIR.glob("*.pdf")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
