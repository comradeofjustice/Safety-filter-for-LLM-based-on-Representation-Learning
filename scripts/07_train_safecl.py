"""Train SafeCL projection head with hierarchical contrastive loss.

The projection head maps frozen Qwen3 embeddings to an intent-aware
safety representation space where:
  - Safe and malicious samples are well-separated
  - Benign-sensitive samples (educational discussion of sensitive topics)
    are pulled towards the safe cluster, not the malicious one

Usage:
  python scripts/07_train_safecl.py
  python scripts/07_train_safecl.py --embed-model qwen3-embedding-8B --epochs 100
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.safecl import SafeCLTrainer, IntentAnnotator
from src.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train SafeCL projection head")
    parser.add_argument(
        "--embed-model",
        default="qwen3-embedding-0.6B",
        help="Embedding model name",
    )
    parser.add_argument(
        "--data-path",
        default="data/processed/train.parquet",
        help="Path to training data (used for intent labels)",
    )
    parser.add_argument(
        "--intent-path",
        default=None,
        help="Path to pre-annotated intent data (if None, annotate from data-path)",
    )
    parser.add_argument(
        "--embeddings-dir",
        default="embeddings",
        help="Base embeddings directory",
    )
    parser.add_argument(
        "--output-dir",
        default="models/safecl",
        help="Base output directory for trained models",
    )
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--output-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(42)

    # Load frozen embeddings
    embed_dir = os.path.join(args.embeddings_dir, args.embed_model)
    train_emb_path = os.path.join(embed_dir, "train.npy")
    train_labels_path = os.path.join(embed_dir, "train_labels.npy")

    logger.info(f"Loading embeddings from {embed_dir}")
    X_train = np.load(train_emb_path)
    y_binary = np.load(train_labels_path)
    logger.info(f"Loaded {X_train.shape[0]} samples, dim={X_train.shape[1]}")

    # Load or create intent labels
    if args.intent_path and os.path.exists(args.intent_path):
        logger.info(f"Loading pre-annotated intent labels from {args.intent_path}")
        df_intent = pd.read_parquet(args.intent_path)
        y_intent = df_intent["intent"].values
    else:
        logger.info("Annotating data with intent labels...")
        df = pd.read_parquet(args.data_path)
        annotator = IntentAnnotator()
        df_annotated = annotator.annotate(df)
        y_intent = df_annotated["intent"].values

        # Print intent distribution
        for intent_id, name in [(0, "safe"), (1, "benign_sensitive"), (2, "malicious")]:
            count = (y_intent == intent_id).sum()
            logger.info(f"  {name}: {count} ({100*count/len(y_intent):.1f}%)")

    # Derive corrected binary labels from intent
    # Benign-sensitive (intent=1) → binary safe (0), not unsafe (1)
    # This is the key innovation: fixing label noise in original safety datasets
    from src.safecl import IntentTaxonomy
    y_binary_corrected = np.array([IntentTaxonomy.to_binary(i) for i in y_intent])
    n_corrected = (y_binary_corrected != y_binary).sum()
    logger.info(f"Corrected {n_corrected} binary labels based on intent "
                f"({100*n_corrected/len(y_binary):.1f}% of data)")

    # Train
    trainer = SafeCLTrainer(
        input_dim=X_train.shape[1],
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        patience=args.patience,
        temperature=args.temperature,
        alpha=args.alpha,
        beta=args.beta,
        device=args.device,
    )

    trainer.train(
        train_embeddings=X_train,
        train_intent_labels=y_intent,
        train_binary_labels=y_binary_corrected,
        finegrained_labels=None,  # Will be added with finer attack-type labels
    )

    # Save
    output_path = os.path.join(args.output_dir, args.embed_model, "projection_head.pkl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    trainer.save(output_path)

    # Generate and save projected training embeddings
    logger.info("Projecting training embeddings...")
    X_train_proj = trainer.project(X_train)
    proj_train_path = os.path.join(args.output_dir, args.embed_model, "train_projected.npy")
    np.save(proj_train_path, X_train_proj)
    logger.info(f"Saved projected training embeddings to {proj_train_path}")

    # Also project and save test embeddings
    test_emb_path = os.path.join(embed_dir, "test.npy")
    if os.path.exists(test_emb_path):
        logger.info("Projecting test embeddings...")
        X_test = np.load(test_emb_path)
        X_test_proj = trainer.project(X_test)
        proj_test_path = os.path.join(args.output_dir, args.embed_model, "test_projected.npy")
        np.save(proj_test_path, X_test_proj)
        logger.info(f"Saved projected test embeddings to {proj_test_path}")


if __name__ == "__main__":
    main()
