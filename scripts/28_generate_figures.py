#!/usr/bin/env python3
"""Generate paper figures: DeepSafe v3 vs SOTA across ALL benchmarks.

Usage:
  python3 scripts/28_generate_figures.py --results valuation/deepsafe_v3_results.json
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJ = Path(__file__).resolve().parent.parent
FIG_DIR = PROJ / "paper" / "figures"


def load_results(results_path):
    with open(results_path) as f:
        return json.load(f)


def plot_accuracy_bars(all_results, output_path):
    """Bar chart: DeepSafe v3 vs top-5 SOTA across all benchmarks (accuracy)."""
    benchmarks = list(all_results.keys())
    n_bench = len(benchmarks)

    # Collect all model names
    all_models = set()
    for bench_results in all_results.values():
        all_models.update(bench_results.keys())
    all_models.discard("DeepSafe-v3")

    # Rank SOTA models by average accuracy
    model_avgs = {}
    for model in all_models:
        accs = []
        for bench_results in all_results.values():
            m = bench_results.get(model, {})
            if "error" not in m:
                accs.append(m.get("accuracy", 0))
        model_avgs[model] = np.mean(accs) if accs else 0

    top_sota = sorted(model_avgs, key=model_avgs.get, reverse=True)[:5]
    all_display_models = ["DeepSafe-v3"] + top_sota

    # Setup plot
    fig, ax = plt.subplots(figsize=(max(12, n_bench * 1.5), 7))
    x = np.arange(n_bench)
    width = 0.8 / len(all_display_models)
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_display_models)))
    colors[0] = [0.9, 0.2, 0.2, 1.0]  # DeepSafe in red

    for i, model in enumerate(all_display_models):
        accs = []
        for bench_results in all_results.values():
            m = bench_results.get(model, {})
            accs.append(m.get("accuracy", 0) if "error" not in m else 0)
        bars = ax.bar(x + i * width, accs, width, label=model, color=colors[i], alpha=0.9)
        # Add value labels
        for bar, acc in zip(bars, accs):
            if acc > 0.5:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{acc:.3f}", ha="center", va="bottom", fontsize=5, rotation=90)

    ax.set_xlabel("Benchmark", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("DeepSafe v3 vs SOTA: Accuracy Across All Benchmarks", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * (len(all_display_models) - 1) / 2)
    ax.set_xticklabels(benchmarks, rotation=30, ha="right", fontsize=9)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(0.4, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=0.8, color="gray", linestyle="--", alpha=0.5, label="80% baseline")
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def plot_radar_chart(all_results, output_path):
    """Radar chart: DeepSafe v3 vs top-3 SOTA across benchmarks."""
    benchmarks = list(all_results.keys())

    # Collect models
    all_models = set()
    for bench_results in all_results.values():
        all_models.update(bench_results.keys())

    # Rank by average accuracy
    model_avgs = {}
    for model in all_models:
        accs = []
        for bench_results in all_results.values():
            m = bench_results.get(model, {})
            if "error" not in m:
                accs.append(m.get("accuracy", 0))
        model_avgs[model] = np.mean(accs) if accs else 0

    display_models = ["DeepSafe-v3"] + sorted(
        [m for m in model_avgs if m != "DeepSafe-v3"],
        key=model_avgs.get, reverse=True
    )[:3]

    # Radar data
    n_vars = len(benchmarks)
    angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
    angles += angles[:1]  # close the circle

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))
    colors = ["#E24A4A", "#4A90D9", "#50B86C", "#F5A623"]

    for i, model in enumerate(display_models):
        values = []
        for bench_results in all_results.values():
            m = bench_results.get(model, {})
            values.append(m.get("accuracy", 0) if "error" not in m else 0)
        values += values[:1]
        ax.fill(angles, values, alpha=0.1, color=colors[i])
        ax.plot(angles, values, "o-", linewidth=2, color=colors[i], label=model, markersize=6)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(benchmarks, fontsize=9)
    ax.set_ylim(0.4, 1.0)
    ax.set_title("Performance Radar: DeepSafe v3 vs SOTA", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def plot_speed_vs_accuracy(all_results, output_path):
    """Scatter: Accuracy vs Inference Time (DeepSafe is fast + accurate)."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Collect all model data points
    points = []  # (name, accuracy, time_s, is_ours)
    for bench_results in all_results.values():
        for model, metrics in bench_results.items():
            if "error" in metrics:
                continue
            acc = metrics.get("accuracy", 0)
            t = metrics.get("inference_time_s", 1)
            is_ours = "DeepSafe" in model
            points.append((model, acc, t, is_ours))

    # Average across benchmarks
    model_data = {}
    for name, acc, t, ours in points:
        if name not in model_data:
            model_data[name] = {"accs": [], "times": [], "ours": ours}
        model_data[name]["accs"].append(acc)
        model_data[name]["times"].append(t)

    for name, data in model_data.items():
        avg_acc = np.mean(data["accs"])
        avg_t = np.mean(data["times"])
        c = "#E24A4A" if data["ours"] else "#4A90D9"
        s = 200 if data["ours"] else 80
        marker = "D" if data["ours"] else "o"
        ax.scatter(avg_t, avg_acc, s=s, c=c, marker=marker, alpha=0.8, edgecolors="black", linewidth=0.5)
        ax.annotate(name, (avg_t, avg_acc), fontsize=7 if not data["ours"] else 10,
                    ha="center", va="bottom", fontweight="bold" if data["ours"] else "normal",
                    xytext=(0, 10), textcoords="offset points")

    ax.set_xlabel("Inference Time (seconds)", fontsize=12)
    ax.set_ylabel("Average Accuracy", fontsize=12)
    ax.set_title("Speed vs Accuracy: DeepSafe v3 vs SOTA Guard Models", fontsize=14, fontweight="bold")
    ax.set_xscale("log")
    ax.grid(alpha=0.3)
    ax.axhline(y=0.8, color="gray", linestyle="--", alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def plot_win_count(all_results, output_path):
    """Horizontal bar chart: Number of benchmarks DeepSafe wins vs each SOTA."""
    benchmarks = list(all_results.keys())
    model_wins = {}

    for bench_results in all_results.values():
        our_acc = bench_results.get("DeepSafe-v3", {}).get("accuracy", 0)
        for model, metrics in bench_results.items():
            if model == "DeepSafe-v3" or "error" in metrics:
                continue
            sota_acc = metrics.get("accuracy", 0)
            if model not in model_wins:
                model_wins[model] = {"wins": 0, "losses": 0, "ties": 0}
            if our_acc > sota_acc + 0.001:
                model_wins[model]["wins"] += 1
            elif abs(our_acc - sota_acc) <= 0.001:
                model_wins[model]["ties"] += 1
            else:
                model_wins[model]["losses"] += 1

    # Sort by wins
    sorted_models = sorted(model_wins.items(), key=lambda x: x[1]["wins"], reverse=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = range(len(sorted_models))
    models = [m for m, _ in sorted_models]
    wins = [d["wins"] for _, d in sorted_models]
    ties = [d["ties"] for _, d in sorted_models]
    losses = [d["losses"] for _, d in sorted_models]
    n_bench = len(benchmarks)

    ax.barh(y_pos, wins, height=0.6, color="#4CAF50", label=f"DeepSafe wins ({sum(wins)})")
    ax.barh(y_pos, ties, height=0.6, left=wins, color="#FFC107", label=f"Ties ({sum(ties)})")
    left_ties = [w + t for w, t in zip(wins, ties)]
    ax.barh(y_pos, losses, height=0.6, left=left_ties, color="#F44336", label=f"SOTA wins ({sum(losses)})")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontsize=9)
    ax.set_xlabel(f"Number of Benchmarks (out of {n_bench})", fontsize=12)
    ax.set_title("DeepSafe v3 Head-to-Head vs SOTA Models", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, n_bench)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="valuation/deepsafe_v3_results.json")
    args = parser.parse_args()

    results_path = PROJ / args.results
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        return

    all_results = load_results(results_path)
    bench_count = len(all_results)
    print(f"Loaded results for {bench_count} benchmarks")

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    plot_accuracy_bars(all_results, FIG_DIR / "fig_benchmark_accuracy.pdf")
    plot_accuracy_bars(all_results, FIG_DIR / "fig_benchmark_accuracy.png")

    if bench_count >= 3:
        plot_radar_chart(all_results, FIG_DIR / "fig_radar.pdf")
        plot_radar_chart(all_results, FIG_DIR / "fig_radar.png")

    plot_speed_vs_accuracy(all_results, FIG_DIR / "fig_speed_vs_accuracy.pdf")
    plot_speed_vs_accuracy(all_results, FIG_DIR / "fig_speed_vs_accuracy.png")

    plot_win_count(all_results, FIG_DIR / "fig_head_to_head.pdf")
    plot_win_count(all_results, FIG_DIR / "fig_head_to_head.png")

    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
