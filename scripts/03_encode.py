#!/usr/bin/env python
"""scripts/03_encode.py - Encode train/test data using Qwen3 Embedding models."""

import argparse
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.encode.qwen_encoder import QwenEncoder
from src.utils.io import load_parquet
from src.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Encode data with Qwen3 Embedding model")
    parser.add_argument(
        "--embed-model",
        type=str,
        required=True,
        help="Embedding model name (e.g., qwen3-embedding-0.6B)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Max sequence length (default: 512)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size (default: 16, auto-reduced to 8 for 8B)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda:0"],
        help="Device to use for encoding (default: auto)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing embeddings",
    )
    args = parser.parse_args()

    set_seed(42)

    embed_dir = f"embeddings/{args.embed_model}"
    os.makedirs(embed_dir, exist_ok=True)

    train_emb_path = os.path.join(embed_dir, "train.npy")
    test_emb_path = os.path.join(embed_dir, "test.npy")
    train_labels_path = os.path.join(embed_dir, "train_labels.npy")
    test_labels_path = os.path.join(embed_dir, "test_labels.npy")

    # Check if already exists
    if (
        not args.overwrite
        and os.path.exists(train_emb_path)
        and os.path.exists(test_emb_path)
    ):
        logger.info(f"Embeddings already exist in {embed_dir}, skipping. Use --overwrite to re-encode.")
        return

    # Load data
    train_df = load_parquet("data/processed/train.parquet")
    test_df = load_parquet("data/processed/test.parquet")

    logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")

    # Initialize encoder
    model_path = f"./pretrained/{args.embed_model}"
    encoder = QwenEncoder(
        model_name_or_path=model_path,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=args.device,
    )

    # Encode train
    logger.info("Encoding training data...")
    train_embeddings = encoder.encode(train_df["text"].tolist())
    np.save(train_emb_path, train_embeddings)
    np.save(train_labels_path, train_df["label"].values)
    logger.info(f"Saved train embeddings: {train_embeddings.shape}")

    # Encode test
    logger.info("Encoding test data...")
    test_embeddings = encoder.encode(test_df["text"].tolist())
    np.save(test_emb_path, test_embeddings)
    np.save(test_labels_path, test_df["label"].values)
    logger.info(f"Saved test embeddings: {test_embeddings.shape}")

    logger.info("Encoding complete!")


if __name__ == "__main__":
    main()
