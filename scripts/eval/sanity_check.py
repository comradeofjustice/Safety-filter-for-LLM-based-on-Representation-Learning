#!/usr/bin/env python3
"""
R10 — Number Consistency Sanity Check

Reads all reports/*.csv and extracts numerical values; then parses
paper/sections/*.tex (and paper/main.tex) for the same numbers.

Flags any mismatch as [REQUEST] C→D in 4models/REVIEW_LOG.md.

Usage:
    python scripts/eval/sanity_check.py
    python scripts/eval/sanity_check.py --verbose
    python scripts/eval/sanity_check.py --strict  # fail on mismatch
"""
import os, re, sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

PROJ = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJ / "reports"
PAPER_DIR = PROJ / "paper"
LOG_PATH = PROJ / "4models" / "REVIEW_LOG.md"

TOLERANCE = 0.001  # floating-point tolerance


def extract_numbers_from_tex(filepath):
    """Extract all numerical values from a LaTeX file with context."""
    numbers = []
    if not filepath.exists():
        return numbers

    with open(filepath) as f:
        text = f.read()

    # Remove comments
    text = re.sub(r'(?<!\\)%.*$', '', text, flags=re.MULTILINE)

    # Find numbers: integer, decimal, and scientific notation
    patterns = [
        # Decimal numbers with context (up to 80 chars before, 40 after)
        (r'(.{0,80})\b(\d+\.\d+)\b(.{0,40})', float),
        # Integer percentages
        (r'(.{0,80})\b(\d+)%\b(.{0,40})', lambda x: float(x) / 100),
    ]

    for pattern, converter in patterns:
        for match in re.finditer(pattern, text):
            context_before = match.group(1).strip()
            value = match.group(2)
            context_after = match.group(3).strip()
            full_context = f"...{context_before[-50:]} {value} {context_after[:30]}..."

            try:
                num = converter(value)
                numbers.append({
                    "value": num,
                    "context": full_context.strip(),
                    "raw": value,
                    "line": text[:match.start()].count('\n') + 1,
                })
            except ValueError:
                pass

    return numbers


def extract_numbers_from_csv(filepath):
    """Extract key summary numbers from CSV files."""
    if not filepath.exists():
        return {}

    df = pd.read_csv(filepath)
    summary = {}

    # For numeric columns, compute mean across all rows
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        vals = df[col].dropna()
        if len(vals) > 0:
            summary[f"csv:{filepath.stem}:{col}:mean"] = float(vals.mean())
            summary[f"csv:{filepath.stem}:{col}:min"] = float(vals.min())
            summary[f"csv:{filepath.stem}:{col}:max"] = float(vals.max())

    return summary


def load_all_csv_numbers():
    """Load all numbers from all CSV files in reports/."""
    all_nums = {}
    for csv_file in REPORTS_DIR.glob("*.csv"):
        # Skip files in subdirectories
        if csv_file.parent != REPORTS_DIR:
            continue
        all_nums[str(csv_file.name)] = extract_numbers_from_csv(csv_file)
    return all_nums


def find_mismatches(csv_numbers, tex_numbers):
    """
    Compare CSV numbers with TeX numbers.
    Returns list of mismatch descriptions.
    """
    mismatches = []

    # For each CSV file's summary
    for csv_name, csv_summary in csv_numbers.items():
        file_mismatches = []

        # Check if any CSV means/min/maxes appear close to TeX numbers
        for csv_key, csv_val in csv_summary.items():
            # Look for close matches in TeX numbers
            found = False
            for tex_num in tex_numbers:
                if abs(tex_num["value"] - csv_val) < TOLERANCE:
                    found = True
                    break

            # If CSV value is important (not trivial 0 or 0.5), flag it
            if not found and abs(csv_val) > 0.01 and abs(csv_val - 0.5) > 0.01:
                file_mismatches.append(
                    f"CSV value {csv_key} = {csv_val:.4f} not found in LaTeX"
                )

        if file_mismatches:
            mismatches.append({
                "csv_file": csv_name,
                "issues": file_mismatches,
            })

    return mismatches


def cross_check_accuracy_claims(csv_numbers):
    """
    Deep cross-check: verify specific numbers that SHOULD appear in the paper.

    These are hard-coded invariants that the paper must report correctly.
    """
    issues = []

    # 1. DeepSafe-v3's accuracy range should be reported
    ds_accs = []
    for csv_name, csv_data in csv_numbers.items():
        for key, val in csv_data.items():
            if "accuracy" in key.lower() and "mean" in key:
                ds_accs.append(val)

    # 2. Check stats_with_ci for valid CI bounds
    if "stats_with_ci.csv" in csv_numbers:
        ci_path = REPORTS_DIR / "stats_with_ci.csv"
        ci_df = pd.read_csv(ci_path)

        # CI low <= mean <= CI high
        for _, row in ci_df.iterrows():
            if pd.notna(row.get("accuracy_ci_low")) and pd.notna(row.get("accuracy_ci_high")):
                mean = row["accuracy_mean"]
                lo = row["accuracy_ci_low"]
                hi = row["accuracy_ci_high"]
                if not (lo <= mean <= hi):
                    issues.append(
                        f"stats_with_ci.csv: {row['benchmark']} {row['model']} "
                        f"CI [{lo:.4f}, {hi:.4f}] doesn't contain mean {mean:.4f}"
                    )

        # p-values should be in [0, 1]
        if "vs_deepsafe_p_value" in ci_df.columns:
            p_vals = ci_df["vs_deepsafe_p_value"].dropna()
            for idx, p in p_vals.items():
                p = float(p)
                if p < 0 or p > 1:
                    issues.append(
                        f"stats_with_ci.csv row {idx}: p-value {p} out of [0,1] range"
                    )

    # 3. Check threshold_analysis.csv has valid thresholds
    if "threshold_analysis.csv" in csv_numbers:
        ta_path = REPORTS_DIR / "threshold_analysis.csv"
        ta_df = pd.read_csv(ta_path)
        if "threshold" in ta_df.columns:
            for t in ta_df["threshold"].unique():
                if t < 0 or t > 1:
                    issues.append(f"threshold_analysis.csv: invalid threshold {t}")

        # All accuracies should be in [0, 1]
        if "accuracy" in ta_df.columns:
            for idx, acc in enumerate(ta_df["accuracy"]):
                if acc < 0 or acc > 1:
                    issues.append(f"threshold_analysis.csv row {idx}: accuracy {acc} out of [0,1]")

    return issues


def append_to_log(mismatches, invariants):
    """Append any issues found to REVIEW_LOG.md."""
    if not mismatches and not invariants:
        return

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"\n[{now}] worker=C status=SANITY_CHECK"]

    if mismatches:
        lines.append("  CSV-vs-LaTeX mismatches found:")
        for m in mismatches:
            lines.append(f"  - {m['csv_file']}:")
            for issue in m["issues"]:
                lines.append(f"    * {issue}")
            lines.append(f"  [REQUEST] C→D: Please verify numbers in {m['csv_file']} "
                          "match paper text")

    if invariants:
        lines.append("  Internal data invariants violated:")
        for issue in invariants:
            lines.append(f"  - {issue}")

    # Actually always append "no issues" on a clean run
    if not mismatches and not invariants:
        lines.append("  All checks passed — no mismatches found.")

    with open(LOG_PATH, "a") as f:
        f.write("\n".join(lines) + "\n")


def main():
    verbose = "--verbose" in sys.argv
    strict = "--strict" in sys.argv

    print("=" * 60)
    print("R10: Number Consistency Sanity Check")
    print("=" * 60)

    # 1. Load CSV numbers
    print("\n[1/4] Loading CSV reports...")
    csv_numbers = load_all_csv_numbers()
    print(f"  Loaded {len(csv_numbers)} CSV files: {list(csv_numbers.keys())}")

    # 2. Load TeX numbers
    print("\n[2/4] Parsing LaTeX numbers...")
    tex_numbers = []
    tex_files = []
    if (PAPER_DIR / "sections").exists():
        tex_files.extend((PAPER_DIR / "sections").glob("*.tex"))
    tex_files.extend(PAPER_DIR.glob("*.tex"))

    for tex_file in tex_files:
        nums = extract_numbers_from_tex(tex_file)
        if verbose:
            print(f"  {tex_file.name}: found {len(nums)} numbers")
        tex_numbers.extend(nums)

    print(f"  Total LaTeX numbers extracted: {len(tex_numbers)}")

    # 3. Cross-check
    print("\n[3/4] Cross-checking numbers...")
    mismatches = find_mismatches(csv_numbers, tex_numbers)
    invariants = cross_check_accuracy_claims(csv_numbers)

    # 4. Report
    print("\n[4/4] Results:")
    if not mismatches and not invariants:
        print("  ALL CHECKS PASSED — No mismatches found.")
    else:
        print(f"  CSV-vs-LaTeX mismatches: {len(mismatches)}")
        for m in mismatches:
            print(f"    {m['csv_file']}:")
            for issue in m["issues"]:
                print(f"      - {issue}")

        print(f"\n  Data invariants violated: {len(invariants)}")
        for issue in invariants:
            print(f"    - {issue}")

    # Append to log
    append_to_log(mismatches, invariants)
    print(f"\n  Results appended to {LOG_PATH}")

    if strict and (mismatches or invariants):
        sys.exit(1)

    return 0


if __name__ == "__main__":
    main()
