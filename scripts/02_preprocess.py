#!/usr/bin/env python
"""scripts/02_preprocess.py - Clean, deduplicate, and split data."""

import argparse
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.io import load_parquet, save_parquet
from src.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Clean text: strip whitespace."""
    if not isinstance(text, str):
        return ""
    return text.strip()


def main():
    parser = argparse.ArgumentParser(description="Preprocess corpus data")
    args = parser.parse_args()

    set_seed(42)

    # Load raw corpus
    raw_path = "data/raw/corpus_raw.parquet"
    if not os.path.exists(raw_path):
        logger.error(f"Raw corpus not found at {raw_path}. Run 01_collect_data.py first.")
        sys.exit(1)

    corpus = load_parquet(raw_path)
    logger.info(f"Loaded {len(corpus)} raw samples")

    # Clean text
    corpus["text"] = corpus["text"].apply(clean_text)

    # Remove empty or invalid text
    initial_count = len(corpus)
    corpus = corpus[corpus["text"].str.len() >= 5]
    corpus = corpus[corpus["text"].str.len() <= 8000]
    logger.info(f"After length filter: {len(corpus)} samples (removed {initial_count - len(corpus)})")

    # Deduplicate by text (keep first occurrence)
    initial_count = len(corpus)
    corpus = corpus.drop_duplicates(subset=["text"], keep="first")
    logger.info(f"After deduplication: {len(corpus)} samples (removed {initial_count - len(corpus)})")

    # Print stats per source
    print("\n=== Dataset Statistics (After Cleaning) ===")
    stats_records = []
    for source in sorted(corpus.source.unique()):
        subset = corpus[corpus.source == source]
        safe = len(subset[subset.label == 0])
        unsafe = len(subset[subset.label == 1])
        total = len(subset)
        stats_records.append({
            "source": source,
            "total": total,
            "safe": safe,
            "unsafe": unsafe,
        })
        print(f"{source:20s}: {total:8d} total  safe={safe:6d}  unsafe={unsafe:6d}")
    print("=" * 60)

    # Save data stats
    stats_df = pd.DataFrame(stats_records)
    os.makedirs("reports", exist_ok=True)
    stats_df.to_csv("reports/data_stats.csv", index=False)
    logger.info("Saved data_stats.csv")

    # Split: train/test = 8:2, stratified by label
    train_df, test_df = train_test_split(
        corpus,
        test_size=0.2,
        stratify=corpus["label"],
        random_state=42,
    )

    logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")

    # Save to data/processed/
    os.makedirs("data/processed", exist_ok=True)
    save_parquet(train_df, "data/processed/train.parquet")
    save_parquet(test_df, "data/processed/test.parquet")

    # Also save combined corpus.parquet
    save_parquet(corpus, "data/processed/corpus.parquet")

    logger.info("Preprocessing complete!")


if __name__ == "__main__":
    main()
