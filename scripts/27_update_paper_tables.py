#!/usr/bin/env python3
"""Generate LaTeX tables from DeepSafe v3 evaluation results for paper update.

Handles all benchmarks dynamically.

Usage:
  python3 scripts/27_update_paper_tables.py --results valuation/deepsafe_v3_results.json
"""

import json
import argparse
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent


def format_row(name, metrics, bold=False):
    """Format a LaTeX table row."""
    if "error" in metrics:
        return f"    {name:35s} & -- & -- & -- \\\\"
    acc = metrics.get("accuracy", 0)
    f1 = metrics.get("f1_macro", 0)
    auc = metrics.get("roc_auc", 0)
    if bold:
        row = (
            "    \\textbf{DeepSafe-v3} & \\textbf{"
            + f"{acc:.4f}"
            + "} & \\textbf{"
            + f"{f1:.4f}"
            + "} & \\textbf{"
            + f"{auc:.4f}"
            + "}"
        )
    else:
        row = f"    {name:35s} & {acc:.4f} & {f1:.4f} & {auc:.4f}"
    return row + " \\\\"


def generate_benchmark_table(all_results, bench_name):
    """Generate a LaTeX table for one benchmark."""
    bench_results = all_results.get(bench_name, {})
    if not bench_results:
        return ""

    # Sort by accuracy
    sorted_models = sorted(
        bench_results.items(),
        key=lambda x: x[1].get("accuracy", 0),
        reverse=True,
    )

    # Separate ours and SOTA
    deepsafe_models = [m for m in sorted_models if "DeepSafe" in m[0]]
    other_models = [m for m in sorted_models if "DeepSafe" not in m[0]]

    lines = []
    lines.append(f"\\caption{{Benchmark results on {bench_name} (30\\% held-out).}}")
    lines.append(f"\\label{{tab:benchmark_{bench_name.lower().replace('-', '').replace(' ', '_')}}}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{@{}lccc@{}}")
    lines.append("\\toprule")
    lines.append("\\textbf{Model} & \\textbf{Accuracy} & \\textbf{F1 (Macro)} & \\textbf{ROC AUC} \\\\")
    lines.append("\\midrule")

    # SOTA models first
    lines.append("\\multicolumn{4}{@{}l}{\\textit{Generative Guard Models (SOTA)}} \\\\")
    for name, metrics in other_models:
        lines.append(format_row(name, metrics))

    # Our models
    lines.append("\\midrule")
    lines.append("\\multicolumn{4}{@{}l}{\\textit{Embedding-Based Models (Ours)}} \\\\")
    for name, metrics in deepsafe_models:
        lines.append(format_row(name, metrics, bold=True))

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    return "\n".join(lines)


def generate_summary_table(all_results):
    """Generate a summary table: DeepSafe v3 vs best SOTA per benchmark."""
    benchmarks = list(all_results.keys())

    lines = []
    lines.append("\\caption{DeepSafe v3 vs. Best SOTA: Per-Benchmark Summary (30\\% held-out data).}")
    lines.append("\\label{tab:sota_v3_summary}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{@{}lccc@{}}")
    lines.append("\\toprule")
    lines.append("\\textbf{Benchmark} & \\textbf{DeepSafe-v3} & \\textbf{Best SOTA} & \\textbf{Winner} \\\\")
    lines.append("\\midrule")

    deepsafe_wins = 0
    sota_wins = 0
    ties = 0

    for bench in benchmarks:
        bench_results = all_results.get(bench, {})
        our_acc = bench_results.get("DeepSafe-v3", {}).get("accuracy", 0)
        sota_models = {k: v for k, v in bench_results.items()
                       if k != "DeepSafe-v3" and "error" not in v}
        if not sota_models:
            continue

        best_sota_name = max(sota_models, key=lambda x: sota_models[x].get("accuracy", 0))
        best_sota_acc = sota_models[best_sota_name]["accuracy"]

        if our_acc > best_sota_acc + 0.001:
            winner = "\\textbf{DeepSafe-v3}"
            deepsafe_wins += 1
        elif abs(our_acc - best_sota_acc) <= 0.001:
            winner = "Tie"
            ties += 1
        else:
            winner = best_sota_name[:20]
            sota_wins += 1

        bname = bench.replace("_", "\\_")
        lines.append(
            f"    {bname:28s} & {our_acc:.4f} & {best_sota_acc:.4f} & {winner} \\\\"
        )

    lines.append("\\midrule")
    lines.append(
        f"    \\multicolumn{{1}}{{l}}{{\\textbf{{Total}}}} & \\multicolumn{{3}}{{l}}{{"
        f"DeepSafe wins: {deepsafe_wins}, SOTA wins: {sota_wins}, Ties: {ties}"
        f"}} \\\\"
    )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    return "\n".join(lines), deepsafe_wins, sota_wins, ties


def generate_wide_summary_table(all_results):
    """Generate a wide table: all models x all benchmarks (accuracy)."""
    benchmarks = list(all_results.keys())

    # Collect all model names across all benchmarks
    all_models = set()
    for bench_results in all_results.values():
        all_models.update(bench_results.keys())

    # Sort: DeepSafe first, then by average accuracy
    model_avgs = {}
    for model in all_models:
        accs = []
        for bench_results in all_results.values():
            m = bench_results.get(model, {})
            if "error" not in m:
                accs.append(m.get("accuracy", 0))
        model_avgs[model] = sum(accs) / max(len(accs), 1)

    sorted_models = sorted(model_avgs, key=model_avgs.get, reverse=True)
    if "DeepSafe-v3" in sorted_models:
        sorted_models.remove("DeepSafe-v3")
    sorted_models = ["DeepSafe-v3"] + sorted_models

    # Build table
    n_bench = len(benchmarks)
    col_spec = "@{}l" + "c" * n_bench + "@{}"
    lines = []
    lines.append("\\caption{DeepSafe v3 vs. SOTA: Accuracy Across All Benchmarks (30\\% held-out).}")
    lines.append("\\label{tab:sota_v3_wide}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{" + col_spec + "}")
    lines.append("\\toprule")
    header = "\\textbf{Model}"
    for b in benchmarks:
        header += f" & \\textbf{{{b[:12]}}}"
    header += " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    for model in sorted_models:
        row = f"    {model:30s}" if model != "DeepSafe-v3" else f"    \\textbf{{DeepSafe-v3:20s}}"
        if model == "DeepSafe-v3":
            row = f"    \\textbf{{DeepSafe-v3}}"
        else:
            row = f"    {model[:30]:30s}"
        for bench in benchmarks:
            metrics = all_results.get(bench, {}).get(model, {})
            if "error" in metrics:
                row += " & --"
            else:
                acc = metrics.get("accuracy", 0)
                if model == "DeepSafe-v3":
                    row += f" & \\textbf{{{acc:.4f}}}"
                else:
                    row += f" & {acc:.4f}"
        row += " \\\\"
        lines.append(row)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="valuation/deepsafe_v3_results.json")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    results_path = PROJ / args.results
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        return

    with open(results_path) as f:
        all_results = json.load(f)

    benchmarks = list(all_results.keys())
    print(f"Loaded results for {len(benchmarks)} benchmarks: {benchmarks}")

    output_lines = []
    output_lines.append("% Auto-generated LaTeX tables from DeepSafe v3 evaluation")
    output_lines.append(f"% Source: {args.results}")
    output_lines.append(f"% Benchmarks: {len(benchmarks)}")
    output_lines.append("")

    # Summary: DeepSafe vs Best SOTA per benchmark
    output_lines.append("% === Per-Benchmark Winner Summary ===")
    output_lines.append("\\begin{table}[t]")
    output_lines.append("\\centering")
    summary_table, dw, sw, tt = generate_summary_table(all_results)
    output_lines.append(summary_table)
    output_lines.append("\\end{table}")
    output_lines.append(f"% Result: DeepSafe wins {dw}/{len(benchmarks)} benchmarks")
    output_lines.append("")

    # Wide summary: all models x all benchmarks
    if len(benchmarks) <= 9:
        output_lines.append("% === Wide Summary: All Models x All Benchmarks ===")
        output_lines.append("\\begin{table}[t]")
        output_lines.append("\\centering")
        output_lines.append(generate_wide_summary_table(all_results))
        output_lines.append("\\end{table}")
        output_lines.append("")

    # Individual benchmark tables (only for benchmarks with >1 model)
    output_lines.append("% === Per-Benchmark Detail Tables ===")
    for bench in benchmarks:
        bench_results = all_results.get(bench, {})
        if len(bench_results) >= 2:  # at least ours + 1 SOTA
            output_lines.append("\\begin{table}[t]")
            output_lines.append("\\centering")
            table = generate_benchmark_table(all_results, bench)
            output_lines.append(table)
            output_lines.append("\\end{table}")
            output_lines.append("")

    latex_output = "\n".join(output_lines)

    if args.output:
        output_path = PROJ / args.output
        output_path.write_text(latex_output)
        print(f"Tables saved to {output_path}")
    else:
        print(latex_output)


if __name__ == "__main__":
    main()
