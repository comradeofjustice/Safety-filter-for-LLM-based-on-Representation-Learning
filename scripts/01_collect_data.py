#!/usr/bin/env python
"""scripts/01_collect_data.py - Collect and load all datasets."""

import argparse
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loaders import LOADERS, log_failure
from src.utils.io import save_parquet
from src.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Collect data from all sources")
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Only load specific dataset (e.g., xguard)",
    )
    args = parser.parse_args()

    set_seed(42)

    # Determine which loaders to use
    if args.only:
        sources = [args.only]
        for s in sources:
            if s not in LOADERS:
                logger.error(f"Unknown source: {s}. Available: {list(LOADERS.keys())}")
                sys.exit(1)
    else:
        sources = list(LOADERS.keys())

    logger.info(f"Loading datasets: {sources}")

    all_records = []

    for source in sources:
        loader_fn = LOADERS[source]
        try:
            df = loader_fn()
            all_records.append(df)
            logger.info(f"✓ Loaded {source}: {len(df)} samples")
        except Exception as e:
            log_failure(source, e)
            logger.warning(f"✗ Failed to load {source}: {e}. Skipping.")
            continue

    if not all_records:
        logger.error("No datasets loaded successfully!")
        sys.exit(1)

    # Combine all datasets
    corpus = pd.concat(all_records, ignore_index=True)
    logger.info(f"Total corpus: {len(corpus)} samples")

    # Save to data/raw/corpus_raw.parquet
    output_path = "data/raw/corpus_raw.parquet"
    save_parquet(corpus, output_path)
    logger.info(f"Saved raw corpus to {output_path}")

    # Print stats per source
    print("\n=== Dataset Statistics ===")
    for source in corpus.source.unique():
        subset = corpus[corpus.source == source]
        safe = len(subset[subset.label == 0])
        unsafe = len(subset[subset.label == 1])
        print(f"{source:20s}: {len(subset):8d} total  safe={safe:6d}  unsafe={unsafe:6d}")
    print("=" * 60)


if __name__ == "__main__":
    import pandas as pd
    main()
