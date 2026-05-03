#!/usr/bin/env python3
"""Create 70/30 held-out splits for ALL benchmark datasets in benchmark/.

Runs on CPU (no GPU needed). Saves held-out 30% for evaluation.
For huge datasets, samples a capped evaluation set.

Datasets:
  1. WildGuardMix - already handled by prepare_augmented_data.py
  2. 100PoisonMpts - already handled
  3. XSTest - already handled
  4. AegisAI-v1 (10,798) - new
  5. AegisAI-v2 (33,416) - partially used; hold out remaining
  6. BeaverTails (330k) - new, sample 3k for eval
  7. DoNotAnswer (939) - new
  8. ToxicChat (10,165) - new
  9. XGuard (200k) - new, sample 3k for eval
  10. CRiskEval (14,888) - new, custom loader needed
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from valuation.run_benchmark import (
    load_aegis1, load_aegis2, load_beavertails,
    load_donotanswer, load_toxicchat, load_xguard,
    load_wildguardmix, load_100poisonmpts, load_xstest,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJ = Path(__file__).resolve().parent.parent
BENCH_DIR = str(PROJ / "benchmark")

# Max evaluation samples for huge datasets (30% capped)
MAX_EVAL_SAMPLES = 5000


def load_criskeval(benchmark_dir, max_samples=None):
    """Load CRiskEval: judge risk based on question.
    Treat high-risk (risk_rank >= 3) as unsafe."""
    import pyarrow.parquet as pq
    path = Path(benchmark_dir) / "CRiskEval" / "data.parquet"
    if not path.exists():
        return None
    df = pq.read_table(path).to_pandas()
    texts = df["question"].tolist()
    # risk_rank: 1-5, higher = more risky. >=3 → unsafe
    labels = (df["risk_rank"] >= 3).astype(int).tolist()
    if max_samples and len(texts) > max_samples:
        idx = np.random.RandomState(42).choice(len(texts), max_samples, replace=False)
        texts = [texts[i] for i in idx]
        labels = [labels[i] for i in idx]
    return {"name": "CRiskEval", "texts": texts, "labels": labels}


BENCHMARKS = [
    # (name, loader, max_eval_samples)
    ("WildGuardMix", lambda: load_wildguardmix(BENCH_DIR), None),
    ("100PoisonMpts", lambda: load_100poisonmpts(BENCH_DIR), None),
    ("XSTest", lambda: load_xstest(BENCH_DIR), None),
    ("AegisAI-v1", lambda: load_aegis1(BENCH_DIR), None),
    ("AegisAI-v2", lambda: load_aegis2(BENCH_DIR), None),
    ("BeaverTails", lambda: load_beavertails(BENCH_DIR), 3000),
    ("DoNotAnswer", lambda: load_donotanswer(BENCH_DIR), None),
    ("ToxicChat", lambda: load_toxicchat(BENCH_DIR), None),
    ("XGuard", lambda: load_xguard(BENCH_DIR), 3000),
    # CRiskEval skipped: ranking task, not binary classification
]


def main():
    np.random.seed(42)
    heldout_dir = PROJ / "valuation" / "heldout"
    heldout_dir.mkdir(parents=True, exist_ok=True)

    # Track which already have held-out
    existing = set()
    for f in heldout_dir.glob("*_30pct.parquet"):
        existing.add(f.stem.replace("_30pct", ""))

    logger.info(f"Existing held-out files: {existing}")
    logger.info("=" * 60)

    for name, loader_fn, max_eval in BENCHMARKS:
        heldout_path = heldout_dir / f"{name}_30pct.parquet"

        # Skip if already exists
        if name in existing or heldout_path.exists():
            logger.info(f"[SKIP] {name}: held-out already exists")
            continue

        data = loader_fn()
        if data is None:
            logger.warning(f"[SKIP] {name}: loader returned None")
            continue

        texts = data["texts"]
        labels = data["labels"]
        n = len(texts)

        # Clean texts
        valid_pairs = [(str(t).strip(), int(l)) for t, l in zip(texts, labels)
                       if str(t).strip() and len(str(t).strip()) >= 3]
        texts = [t for t, l in valid_pairs]
        labels = [l for t, l in valid_pairs]
        n = len(texts)
        n_safe = sum(1 for l in labels if l == 0)
        n_unsafe = sum(1 for l in labels if l == 1)

        logger.info(f"{name}: total={n} (safe={n_safe}, unsafe={n_unsafe})")

        # Split 70/30
        n_held = int(n * 0.3)
        if max_eval and n_held > max_eval:
            n_held = max_eval

        indices = np.random.permutation(n)
        held_idx = indices[:n_held]

        held_texts = [texts[i] for i in held_idx]
        held_labels = [labels[i] for i in held_idx]

        n_held_safe = sum(1 for l in held_labels if l == 0)
        n_held_unsafe = sum(1 for l in held_labels if l == 1)

        # Save held-out
        pd.DataFrame({"texts": held_texts, "labels": held_labels}).to_parquet(heldout_path)
        logger.info(f"  → Saved {n_held} held-out samples (safe={n_held_safe}, unsafe={n_held_unsafe})")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("ALL HELD-OUT FILES:")
    total_texts = 0
    for f in sorted(heldout_dir.glob("*_30pct.parquet")):
        df = pd.read_parquet(f)
        n_s = (df["labels"] == 0).sum()
        n_u = (df["labels"] == 1).sum()
        name = f.stem.replace("_30pct", "")
        logger.info(f"  {name:25s}: {len(df):6d} samples (safe={n_s}, unsafe={n_u})")
        total_texts += len(df)
    logger.info(f"  {'TOTAL':25s}: {total_texts:6d} samples")
    logger.info("Done!")


if __name__ == "__main__":
    main()
