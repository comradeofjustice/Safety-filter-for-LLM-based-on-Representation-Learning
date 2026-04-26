#!/usr/bin/env python
"""Encode train/test data via vLLM embedding API."""

import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.encode.vllm_encoder import VLLMEncoder
from src.utils.io import load_parquet
from src.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed-model", type=str, required=True)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    set_seed(42)

    embed_dir = f"embeddings/{args.embed_model}"
    os.makedirs(embed_dir, exist_ok=True)

    train_emb_path = os.path.join(embed_dir, "train.npy")
    test_emb_path = os.path.join(embed_dir, "test.npy")

    if not args.overwrite and os.path.exists(train_emb_path) and os.path.exists(test_emb_path):
        logger.info(f"Embeddings already exist in {embed_dir}, skipping.")
        return

    train_df = load_parquet("data/processed/train.parquet")
    test_df = load_parquet("data/processed/test.parquet")
    logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")

    encoder = VLLMEncoder(
        base_url=f"http://localhost:{args.port}",
        model_name=args.embed_model,
        batch_size=args.batch_size,
    )

    logger.info("Encoding training data...")
    train_emb = encoder.encode(train_df["text"].tolist())
    np.save(train_emb_path, train_emb)
    np.save(os.path.join(embed_dir, "train_labels.npy"), train_df["label"].values)
    logger.info(f"Saved train embeddings: {train_emb.shape}")

    logger.info("Encoding test data...")
    test_emb = encoder.encode(test_df["text"].tolist())
    np.save(test_emb_path, test_emb)
    np.save(os.path.join(embed_dir, "test_labels.npy"), test_df["label"].values)
    logger.info(f"Saved test embeddings: {test_emb.shape}")

    logger.info("Encoding complete!")


if __name__ == "__main__":
    main()
