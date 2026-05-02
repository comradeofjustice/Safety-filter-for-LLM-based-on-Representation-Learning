#!/usr/bin/env python3
"""Train DeepSafe: Advanced hyperbolic projection head for safety classification.

NeurIPS-level innovations:
  1. Hyperbolic geometry (Poincaré ball) for hierarchical safety taxonomy
  2. Optimal Transport (Sinkhorn) class regularization
  3. Learnable prototype anchors with hierarchical margins
  4. Spectral decorrelation (Barlow Twins) for feature diversity
  5. Counterfactual pair repulsion for intent awareness

Usage:
  python scripts/20_train_deepsafe.py
  python scripts/20_train_deepsafe.py --embed-model qwen3-embedding-8B
  python scripts/20_train_deepsafe.py --embed-model qwen3-embedding-0.6B --epochs 50
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.deepsafe.trainer import DeepSafeTrainer
from src.deepsafe.projection_head import DeepSafeProjectionHeadManager
from src.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_counterfactual_pairs(pairs_path: str):
    """Load counterfactual pair embeddings for training."""
    if not os.path.exists(pairs_path):
        logger.warning(f"No counterfactual pairs found at {pairs_path}")
        return None

    try:
        pairs = pd.read_parquet(pairs_path)
        logger.info(f"Loaded {len(pairs)} counterfactual pairs")

        # Encode the pairs using the same embedding model
        # For now, we use pre-computed pairs
        return pairs
    except Exception as e:
        logger.warning(f"Failed to load counterfactual pairs: {e}")
        return None


def generate_cf_embeddings(cf_pairs, encoder, device="cuda"):
    """Generate embeddings for counterfactual pairs."""
    malicious_texts = cf_pairs["malicious"].tolist()
    benign_texts = cf_pairs["benign"].tolist()

    logger.info(f"Encoding {len(malicious_texts)} counterfactual pairs...")
    emb_mal = encoder.encode(malicious_texts)
    emb_ben = encoder.encode(benign_texts)

    return emb_mal, emb_ben


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed-model", default="qwen3-embedding-0.6B")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--output-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--lambda-ot", type=float, default=0.3)
    parser.add_argument("--lambda-proto", type=float, default=0.2)
    parser.add_argument("--lambda-decorr", type=float, default=0.005)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-cf", action="store_true", help="Skip counterfactual pairs")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--tag", default="v1")
    args = parser.parse_args()

    set_seed(42)

    device = args.device if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.empty_cache()
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}, Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    # Paths
    embed_dir = f"embeddings/{args.embed_model}"
    output_dir = args.output_dir or f"models/deepsafe_{args.tag}"
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    logger.info("=" * 60)
    logger.info("Loading training data...")
    logger.info("=" * 60)

    X_train = np.load(os.path.join(embed_dir, "train.npy"))
    y_train = np.load(os.path.join(embed_dir, "train_labels.npy"))

    # Load intent labels
    intent_path = "data/processed/train_intent.parquet"
    if os.path.exists(intent_path):
        intent_df = pd.read_parquet(intent_path)
        y_train_intent = intent_df["intent"].values
        logger.info(f"Loaded intent labels: safe={(y_train_intent==0).sum()}, "
                    f"benign={(y_train_intent==1).sum()}, malicious={(y_train_intent==2).sum()}")
    else:
        logger.warning("No intent labels found, using binary labels")
        y_train_intent = y_train.copy()

    # Apply label correction (benign_sensitive -> safe)
    y_train_corrected = y_train.copy()
    correction_mask = y_train_intent == 1  # benign_sensitive
    y_train_corrected[correction_mask] = 0
    n_corrected = correction_mask.sum()
    logger.info(f"Label correction: {n_corrected} benign-sensitive samples re-labeled as safe")

    logger.info(f"Training: {X_train.shape[0]} samples, {X_train.shape[1]} dims")

    # Load counterfactual pairs
    cf_pairs = None
    if not args.no_cf:
        cf_path = "data/processed/contrastive_pairs.parquet"
        pairs = load_counterfactual_pairs(cf_path)
        if pairs is not None:
            try:
                from src.encode.qwen_encoder import QwenEncoder
                model_path = f"./pretrained/{args.embed_model}"
                # Use CPU for encoding CF pairs to save GPU memory
                encoder = QwenEncoder(model_path, device="cpu", batch_size=8)
                emb_mal, emb_ben = generate_cf_embeddings(pairs, encoder, device="cpu")
                cf_pairs = (emb_mal, emb_ben)
                logger.info(f"Encoded CF pairs: mal={emb_mal.shape}, ben={emb_ben.shape}")
                del encoder
            except Exception as e:
                logger.warning(f"Could not encode CF pairs: {e}")

    # Train DeepSafe
    logger.info("=" * 60)
    logger.info("Training DeepSafe...")
    logger.info("=" * 60)

    start_time = time.time()

    trainer = DeepSafeTrainer(
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
        lambda_ot=args.lambda_ot,
        lambda_proto=args.lambda_proto,
        lambda_decorr=args.lambda_decorr,
        gamma=args.gamma,
        device=device,
    )

    trainer.train(
        X_train,
        y_train_intent,
        y_train_corrected,
        cf_pairs=cf_pairs,
    )

    elapsed = time.time() - start_time
    logger.info(f"Training completed in {elapsed:.1f}s ({elapsed/60:.1f}m)")

    # Save model
    proj_path = os.path.join(output_dir, args.embed_model, "projection_head.pkl")
    os.makedirs(os.path.dirname(proj_path), exist_ok=True)
    trainer.save(proj_path)

    # Save config
    config = {
        "timestamp": datetime.now().isoformat(),
        "embed_model": args.embed_model,
        "input_dim": X_train.shape[1],
        "output_dim": args.output_dim,
        "hidden_dim": args.hidden_dim,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "best_epoch": trainer.best_epoch + 1,
        "best_val_loss": float(trainer.best_loss),
        "train_time_s": elapsed,
        "n_train_samples": len(X_train),
        "n_label_corrections": int(n_corrected),
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Model saved to {proj_path}")
    logger.info(f"Config saved to {output_dir}/config.json")

    # Clean up GPU
    if device == "cuda":
        del trainer
        torch.cuda.empty_cache()

    logger.info("DeepSafe training complete!")


if __name__ == "__main__":
    main()
