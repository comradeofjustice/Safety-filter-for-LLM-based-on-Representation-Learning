#!/usr/bin/env python3
"""
R5 — Bootstrap Confidence Intervals & Significance Testing

For DeepSafe-v3 (5-seed results):
  - Paired bootstrap (N=10000) for 95% CI on accuracy/F1/AUC
  - Per-benchmark mean, std, ci_low, ci_high

For DeepSafe-v3 vs each SOTA/baseline model:
  - McNemar's test on instance-level binary predictions (paired, same samples)
  - Reports p-value
  - Marks "comparable" when p > 0.05 or diff < 2σ

Output: reports/stats_with_ci.csv
"""
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

PROJ = Path(__file__).resolve().parents[2]
PRED_DIR = PROJ / "valuation" / "heldout" / "predictions"
HELDOUT_DIR = PROJ / "valuation" / "heldout"
OUTPUT_PATH = PROJ / "reports" / "stats_with_ci.csv"

N_BOOTSTRAP = 10000
CI_ALPHA = 0.05  # 95% CI
SEEDS = ["seed_42", "seed_0", "seed_1", "seed_2", "seed_3"]

BENCHMARK_PARQUET_MAP = {
    "WildGuardMix": "WildGuardMix_30pct.parquet",
    "XSTest": "XSTest_30pct.parquet",
    "100PoisonMpts": "100PoisonMpts_30pct.parquet",
    "AegisAI-v1": "AegisAI-v1_30pct.parquet",
    "AegisAI-v2": "AegisAI-v2_30pct.parquet",
    "BeaverTails": "BeaverTails_30pct.parquet",
    "DoNotAnswer": "DoNotAnswer_30pct.parquet",
    "ToxicChat": "ToxicChat_30pct.parquet",
    "XGuard": "XGuard_30pct.parquet",
}


def load_ground_truth(benchmark_name):
    parquet_name = BENCHMARK_PARQUET_MAP.get(benchmark_name)
    if parquet_name is None:
        return None
    path = HELDOUT_DIR / parquet_name
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return df["labels"].values.astype(int)


def paired_bootstrap_ci(y_true, y_scores_list, n_bootstrap=N_BOOTSTRAP, alpha=CI_ALPHA):
    """
    Compute bootstrap CI for the MEAN metric across seeds.

    For each bootstrap iteration:
      1. Resample test samples with replacement (paired across seeds)
      2. For each seed, compute the metric on bootstrapped samples
      3. Average metrics across seeds
      4. This gives the bootstrap distribution of mean accuracy/F1/AUC

    y_scores_list: list of (y_pred_binary, y_score) tuples from each seed
    Returns: (mean_acc, std_acc, ci_low_acc, ci_high_acc,
              mean_f1,  std_f1,  ci_low_f1,  ci_high_f1,
              mean_auc, std_auc, ci_low_auc, ci_high_auc)
    """
    rng = np.random.RandomState(42)
    n_seeds = len(y_scores_list)
    n_samples = len(y_true)
    n_classes = len(np.unique(y_true))

    boot_accs = []
    boot_f1s = []
    boot_aucs = []

    for _ in range(n_bootstrap):
        sample_idx = rng.randint(0, n_samples, n_samples)

        iter_accs = []
        iter_f1s = []
        iter_aucs = []

        for y_pred, y_score in y_scores_list:
            y_pred_boot = y_pred[sample_idx]
            y_score_boot = y_score[sample_idx]
            y_true_boot = y_true[sample_idx]

            iter_accs.append(accuracy_score(y_true_boot, y_pred_boot))
            if n_classes >= 2:
                iter_f1s.append(f1_score(y_true_boot, y_pred_boot, average="macro"))
                try:
                    iter_aucs.append(roc_auc_score(y_true_boot, y_score_boot))
                except ValueError:
                    iter_aucs.append(0.5)
            else:
                iter_f1s.append(0.0)
                iter_aucs.append(0.5)

        boot_accs.append(np.mean(iter_accs))
        boot_f1s.append(np.mean(iter_f1s))
        boot_aucs.append(np.mean(iter_aucs))

    boot_accs = np.array(boot_accs)
    boot_f1s = np.array(boot_f1s)
    boot_aucs = np.array(boot_aucs)

    def ci(arr):
        lo = np.percentile(arr, 100 * alpha / 2)
        hi = np.percentile(arr, 100 * (1 - alpha / 2))
        return np.mean(arr), np.std(arr), lo, hi

    return (*ci(boot_accs), *ci(boot_f1s), *ci(boot_aucs))


def mcnemar_test(y_true, y_pred_a, y_pred_b):
    """
    McNemar's test for paired binary predictions.
    Tests whether model A and model B have the same error rate.
    """
    # b: A wrong, B correct
    # c: A correct, B wrong
    a_correct = y_pred_a == y_true
    b_correct = y_pred_b == y_true
    b = ((~a_correct) & b_correct).sum()
    c = (a_correct & (~b_correct)).sum()

    if b + c == 0:
        return 1.0  # perfect agreement

    # McNemar statistic with continuity correction
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - sp_stats.chi2.cdf(stat, 1)
    return p_value


def main():
    print("=" * 70)
    print("R5: Bootstrap CI & Significance Testing")
    print("=" * 70)

    # 1. Load DeepSafe-v3 per-seed predictions
    ds_preds = {}  # benchmark -> [(y_pred, y_score), ...] per seed
    v31_dir = PRED_DIR / "v3.1"

    gt_cache = {}

    for benchmark_name in BENCHMARK_PARQUET_MAP:
        # Load ground truth
        y_true = load_ground_truth(benchmark_name)
        if y_true is None:
            continue
        gt_cache[benchmark_name] = y_true

        seed_data = []
        for seed in SEEDS:
            npy_path = v31_dir / seed / f"{benchmark_name}.npy"
            if not npy_path.exists():
                continue
            y_score = np.load(npy_path)
            n = min(len(y_true), len(y_score))
            y_score_n = y_score[:n]
            y_pred_n = (y_score_n > 0.5).astype(int)
            seed_data.append((y_pred_n, y_score_n))

        if len(seed_data) >= 3:  # need at least 3 seeds for meaningful CI
            ds_preds[benchmark_name] = (y_true[:n], seed_data)

    print(f"DeepSafe-v3: {len(ds_preds)} benchmarks with multi-seed data")

    # 2. Load SOTA and baseline predictions
    other_models = {}  # benchmark -> {model_name: (y_pred, y_score)}

    for model_type in ["sota", "baselines"]:
        type_dir = PRED_DIR / model_type
        if not type_dir.exists():
            continue
        for model_dir in sorted(type_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model_name = model_dir.name
            for npy_file in model_dir.glob("*.npy"):
                benchmark = npy_file.stem
                if benchmark not in gt_cache:
                    continue
                y_true = gt_cache[benchmark]
                y_score = np.load(str(npy_file))
                n = min(len(y_true), len(y_score))
                y_score_n = y_score[:n]
                y_pred_n = (y_score_n > 0.5).astype(int)

                if benchmark not in other_models:
                    other_models[benchmark] = {}
                other_models[benchmark][model_name] = (y_pred_n, y_score_n)

    # 3. Bootstrap CI for DeepSafe-v3
    rows = []

    for benchmark in sorted(ds_preds):
        y_true, seed_data = ds_preds[benchmark]
        y_true_n = y_true[:len(seed_data[0][0])]
        n_classes = len(np.unique(y_true_n))

        # Point estimates per seed
        seed_metrics = []
        for y_pred, y_score in seed_data:
            yt = y_true_n[:len(y_pred)]
            acc = accuracy_score(yt, y_pred)
            f1 = f1_score(yt, y_pred, average="macro") if n_classes >= 2 else 0.0
            try:
                auc = roc_auc_score(yt, y_score) if n_classes >= 2 else 0.5
            except ValueError:
                auc = 0.5
            seed_metrics.append((acc, f1, auc))

        mean_acc = np.mean([m[0] for m in seed_metrics])
        std_acc = np.std([m[0] for m in seed_metrics], ddof=1)
        mean_f1 = np.mean([m[1] for m in seed_metrics])
        std_f1 = np.std([m[1] for m in seed_metrics], ddof=1)
        mean_auc = np.mean([m[2] for m in seed_metrics])
        std_auc = np.std([m[2] for m in seed_metrics], ddof=1)

        # Bootstrap CI
        (boot_acc, boot_acc_std, acc_lo, acc_hi,
         boot_f1, boot_f1_std, f1_lo, f1_hi,
         boot_auc, boot_auc_std, auc_lo, auc_hi) = paired_bootstrap_ci(y_true_n, seed_data)

        rows.append({
            "benchmark": benchmark,
            "model": "DeepSafe-v3",
            "n_seeds": len(seed_data),
            "accuracy_mean": round(mean_acc, 4),
            "accuracy_std": round(std_acc, 4),
            "accuracy_ci_low": round(acc_lo, 4),
            "accuracy_ci_high": round(acc_hi, 4),
            "f1_mean": round(mean_f1, 4),
            "f1_std": round(std_f1, 4),
            "f1_ci_low": round(f1_lo, 4),
            "f1_ci_high": round(f1_hi, 4),
            "auc_mean": round(mean_auc, 4),
            "auc_std": round(std_auc, 4),
            "auc_ci_low": round(auc_lo, 4),
            "auc_ci_high": round(auc_hi, 4),
            "vs_deepsafe_p_value": "",
            "is_comparable": "",
            "test_type": "",
        })

        # 4. McNemar test: DeepSafe vs each other model
        if benchmark not in other_models:
            continue

        # DeepSafe ensemble prediction (majority or avg across seeds)
        ds_ensemble_scores = np.mean([sd[1] for sd in seed_data], axis=0)
        ds_ensemble_preds = (ds_ensemble_scores > 0.5).astype(int)

        for other_model, (other_preds, other_scores) in other_models[benchmark].items():
            yt = y_true_n[:len(other_preds)]
            ds_p = ds_ensemble_preds[:len(other_preds)]
            ot_p = other_preds[:len(other_preds)]

            # McNemar test
            mcnemar_p = mcnemar_test(yt, ds_p, ot_p)

            # Per-metric comparison
            ot_acc = accuracy_score(yt, ot_p)
            ot_f1 = f1_score(yt, ot_p, average="macro") if n_classes >= 2 else 0.0
            try:
                ot_auc = roc_auc_score(yt, other_scores[:len(yt)]) if n_classes >= 2 else 0.5
            except ValueError:
                ot_auc = 0.5

            # Check if "comparable": p > 0.05 or difference < 2 * std
            acc_diff = abs(mean_acc - ot_acc)
            comparable = "comparable" if (mcnemar_p > 0.05 or acc_diff < 2 * max(std_acc, 0.001)) else ""

            rows.append({
                "benchmark": benchmark,
                "model": other_model,
                "n_seeds": 1,
                "accuracy_mean": round(ot_acc, 4),
                "accuracy_std": 0.0,
                "accuracy_ci_low": "",
                "accuracy_ci_high": "",
                "f1_mean": round(ot_f1, 4),
                "f1_std": 0.0,
                "f1_ci_low": "",
                "f1_ci_high": "",
                "auc_mean": round(ot_auc, 4),
                "auc_std": 0.0,
                "auc_ci_low": "",
                "auc_ci_high": "",
                "vs_deepsafe_p_value": round(mcnemar_p, 6),
                "is_comparable": comparable,
                "test_type": "McNemar",
            })

    # 5. Save
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False, float_format="%.4f")
    print(f"\nSaved to {OUTPUT_PATH}")
    print(f"  Total rows: {len(df)}")

    # 6. Summary
    print(f"\n{'='*70}")
    print("DeepSafe-v3 Bootstrap CI Summary (accuracy):")
    print(f"{'='*70}")
    print(f"{'Benchmark':<20} {'Mean':>8} {'Std':>8} {'CI Low':>8} {'CI High':>8}")
    print("-" * 56)
    ds_rows = df[df["model"] == "DeepSafe-v3"]
    for _, r in ds_rows.iterrows():
        print(f"{r['benchmark']:<20} {r['accuracy_mean']:>8.4f} {r['accuracy_std']:>8.4f} "
              f"{r['accuracy_ci_low']:>8.4f} {r['accuracy_ci_high']:>8.4f}")

    print(f"\nSignificance (DeepSafe-v3 vs others, McNemar p-value):")
    comparable_count = 0
    sig_rows = df[df["model"] != "DeepSafe-v3"]
    for _, r in sig_rows.iterrows():
        marker = "  ← comparable" if r["is_comparable"] else ""
        if r["is_comparable"]:
            comparable_count += 1
        print(f"  {r['benchmark']:<20} vs {r['model']:<35} p={r['vs_deepsafe_p_value']:.6f}{marker}")
    print(f"\n  {comparable_count}/{len(sig_rows)} comparisons marked 'comparable'")


if __name__ == "__main__":
    main()
