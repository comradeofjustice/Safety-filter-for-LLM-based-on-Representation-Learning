"""Annotate training data with 3-class intent labels.

Intent taxonomy:
  0 - Safe: Clearly benign content
  1 - Benign-Sensitive: Discussion of sensitive topics with educational intent
  2 - Malicious: Actually harmful intent

Usage:
  python scripts/06_annotate_intent.py
  python scripts/06_annotate_intent.py --use-llm --llm-model gpt-4
"""

import argparse
import logging
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.safecl.intent import IntentAnnotator, IntentTaxonomy
from src.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Annotate data with intent labels")
    parser.add_argument(
        "--data-path",
        default="data/processed/train.parquet",
        help="Path to training data",
    )
    parser.add_argument(
        "--output-path",
        default="data/processed/train_intent.parquet",
        help="Path to save annotated data",
    )
    parser.add_argument(
        "--use-llm", action="store_true", help="Use LLM for annotation (requires API key)"
    )
    parser.add_argument("--llm-model", default=None, help="LLM model name")
    parser.add_argument(
        "--sample-size", type=int, default=None, help="Sample size for LLM annotation"
    )
    args = parser.parse_args()

    set_seed(42)

    # Load data
    logger.info(f"Loading training data from {args.data_path}")
    df = pd.read_parquet(args.data_path)
    logger.info(f"Loaded {len(df)} samples")

    # Annotate
    annotator = IntentAnnotator(use_llm=args.use_llm, llm_model=args.llm_model)
    df = annotator.annotate(df)

    # Print stats
    logger.info("Intent distribution:")
    for name, intent_id in [("safe", 0), ("benign_sensitive", 1), ("malicious", 2)]:
        count = (df["intent"] == intent_id).sum()
        pct = 100 * count / len(df)
        logger.info(f"  {name}: {count} ({pct:.1f}%)")

    # Cross-tabulation with binary labels
    logger.info("\nCross-tabulation (binary label vs intent):")
    ct = pd.crosstab(df["label"], df["intent"])
    ct.index = ["safe (0)", "unsafe (1)"]
    ct.columns = ["safe (0)", "benign_sensitive (1)", "malicious (2)"]
    logger.info(f"\n{ct}")

    # Cross-tabulation with source
    logger.info("\nSource distribution by intent:")
    for source in df["source"].unique():
        source_df = df[df["source"] == source]
        counts = {}
        for intent_id, intent_name in IntentTaxonomy.names().items():
            counts[intent_name] = (source_df["intent"] == intent_id).sum()
        logger.info(f"  {source}: {counts}")

    # Save
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    df.to_parquet(args.output_path, index=False)
    logger.info(f"Saved annotated data to {args.output_path}")

    # Also save intent distribution summary
    stats_path = args.output_path.replace(".parquet", "_stats.csv")
    pd.DataFrame(
        {
            "intent": ["safe", "benign_sensitive", "malicious"],
            "count": [
                (df["intent"] == 0).sum(),
                (df["intent"] == 1).sum(),
                (df["intent"] == 2).sum(),
            ],
        }
    ).to_csv(stats_path, index=False)
    logger.info(f"Saved intent stats to {stats_path}")


if __name__ == "__main__":
    main()
