#!/usr/bin/env python3
"""Train robust projection head with advanced losses (no hyperbolic components).

Simpler architecture: 2-layer MLP + residual + spectral norm
Combined with: hierarchical contrastive + OT + prototype + decorrelation loss

Usage:
  python scripts/23_train_robust.py --embed-model qwen3-embedding-8B --tag v1
  python scripts/23_train_robust.py --embed-model qwen3-embedding-0.6B --tag v1
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
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.deepsafe.robust_projection_head import RobustProjectionHead, RobustHeadManager
from src.deepsafe.losses import DeepSafeLoss
from src.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_robust(
    X_train, y_train_intent, y_train_binary,
    input_dim, hidden_dim=512, output_dim=256, dropout=0.1,
    lr=1e-3, weight_decay=1e-4, batch_size=256,
    max_epochs=100, patience=15,
    temperature=0.07, alpha=2.0,
    lambda_ot=0.3, lambda_proto=0.2, lambda_decorr=0.005,
    device="cuda",
):
    """Train RobustProjectionHead with DeepSafe loss."""
    # Split train/val
    indices = np.arange(len(X_train))
    tr_idx, val_idx = train_test_split(
        indices, test_size=0.1, stratify=y_train_intent, random_state=42
    )

    X_tr = X_train[tr_idx]
    X_val = X_train[val_idx]
    y_tr_intent = y_train_intent[tr_idx]
    y_val_intent = y_train_intent[val_idx]
    y_tr_binary = y_train_binary[tr_idx]
    y_val_binary = y_train_binary[val_idx]

    # Build dataloader
    train_dataset = TensorDataset(
        torch.FloatTensor(X_tr), torch.LongTensor(y_tr_intent), torch.LongTensor(y_tr_binary)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_intent_t = torch.LongTensor(y_val_intent).to(device)
    y_val_binary_t = torch.LongTensor(y_val_binary).to(device)

    # Build model
    model = RobustProjectionHead(
        input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout=dropout
    ).to(device)

    criterion = DeepSafeLoss(
        feature_dim=output_dim, temperature=temperature, alpha=alpha,
        lambda_ot=lambda_ot, lambda_proto=lambda_proto, lambda_decorr=lambda_decorr,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_state = None

    for epoch in range(max_epochs):
        # Training
        model.train()
        epoch_losses = {"total": 0, "hierarchical": 0, "binary": 0, "intent": 0}
        n_batches = 0

        for batch in train_loader:
            batch_x = batch[0].to(device)
            batch_intent = batch[1].to(device)
            batch_binary = batch[2].to(device)

            optimizer.zero_grad()
            features = model(batch_x)
            loss, loss_components = criterion(features, batch_binary, batch_intent, return_components=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            for k in epoch_losses:
                if k in loss_components:
                    epoch_losses[k] += loss_components[k]
            n_batches += 1

        for k in epoch_losses:
            epoch_losses[k] /= max(n_batches, 1)

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_features = model(X_val_t)
            val_loss, val_comps = criterion(val_features, y_val_binary_t, y_val_intent_t, return_components=True)

        current_lr = scheduler.get_last_lr()[0]
        logger.info(
            f"Epoch {epoch+1}/{max_epochs}: train={epoch_losses['total']:.4f}, "
            f"val={val_comps['total']:.4f}, hier={epoch_losses.get('hierarchical', 0):.4f}, "
            f"ot={val_comps.get('sinkhorn_ot', 0):.4f}, lr={current_lr:.2e}"
        )

        # Early stopping
        val_total = val_comps["total"]
        if val_total < best_val_loss:
            best_val_loss = val_total
            best_epoch = epoch
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}, best was epoch {best_epoch+1}")
                break

        torch.cuda.empty_cache()

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)

    logger.info(f"Training complete. Best val_loss={best_val_loss:.4f} at epoch {best_epoch+1}")

    return model, best_val_loss, best_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed-model", default="qwen3-embedding-8B")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--output-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--lambda-ot", type=float, default=0.3)
    parser.add_argument("--lambda-proto", type=float, default=0.2)
    parser.add_argument("--lambda-decorr", type=float, default=0.005)
    parser.add_argument("--tag", default="v1")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    set_seed(42)

    device = args.device if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.empty_cache()

    embed_dir = f"embeddings/{args.embed_model}"
    output_dir = f"models/robust_{args.tag}"
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    logger.info("Loading data...")
    X_train = np.load(os.path.join(embed_dir, "train.npy"))
    y_train = np.load(os.path.join(embed_dir, "train_labels.npy"))

    intent_path = "data/processed/train_intent.parquet"
    if os.path.exists(intent_path):
        intent_df = pd.read_parquet(intent_path)
        y_train_intent = intent_df["intent"].values
    else:
        y_train_intent = y_train.copy()

    # Label correction
    y_train_corrected = y_train.copy()
    y_train_corrected[y_train_intent == 1] = 0
    logger.info(f"Corrected {(y_train_intent == 1).sum()} benign-sensitive labels")

    logger.info(f"Training on {X_train.shape[0]} samples, dim={X_train.shape[1]}")

    start_time = time.time()

    model, best_loss, best_epoch = train_robust(
        X_train=X_train,
        y_train_intent=y_train_intent,
        y_train_binary=y_train_corrected,
        input_dim=X_train.shape[1],
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        lr=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        alpha=args.alpha,
        lambda_ot=args.lambda_ot,
        lambda_proto=args.lambda_proto,
        lambda_decorr=args.lambda_decorr,
        device=device,
    )

    elapsed = time.time() - start_time

    # Save
    proj_path = os.path.join(output_dir, args.embed_model, "projection_head.pkl")
    os.makedirs(os.path.dirname(proj_path), exist_ok=True)
    RobustHeadManager.save(model, proj_path)

    config = {
        "timestamp": datetime.now().isoformat(),
        "embed_model": args.embed_model,
        "input_dim": X_train.shape[1],
        "output_dim": args.output_dim,
        "hidden_dim": args.hidden_dim,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs_trained": best_epoch + 1,
        "best_val_loss": float(best_loss),
        "train_time_s": elapsed,
        "params": sum(p.numel() for p in model.parameters()),
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Model saved to {proj_path}")
    logger.info(f"Training time: {elapsed:.1f}s ({elapsed/60:.1f}m)")
    logger.info(f"Parameters: {config['params']:,}")

    # Clean up
    if device == "cuda":
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
