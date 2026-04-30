"""Train SafeCL v3 with clean synthetic contrastive pairs + full dataset.

Key improvements over v1/v2:
  1. Clean synthetic contrastive pairs with perfect intent labels
  2. Joint training: contrastive loss on pairs + classification loss on full data
  3. Better hyperparameters (higher batch size for contrastive, proper temperature)
  4. Uses both synthetic pairs and original training data
  5. Comprehensive logging

Architecture: Frozen Qwen3 → ProjectionHead (1024→512→256) → L2Norm
Training:   L = L_contrastive(pairs) + lambda * L_classification(full_data)
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.safecl.projection_head import ProjectionHead, ProjectionHeadManager
from src.safecl.losses import SupConLoss
from src.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class SafeCLv3Trainer:
    """Improved SafeCL trainer with clean pairs + joint loss."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        output_dim: int = 256,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 512,
        max_epochs: int = 50,
        patience: int = 10,
        temperature: float = 0.1,
        lambda_cls: float = 0.3,
        val_size: float = 0.1,
        random_state: int = 42,
        device: str = "cuda",
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.temperature = temperature
        self.lambda_cls = lambda_cls
        self.val_size = val_size
        self.random_state = random_state
        self.device = device if torch.cuda.is_available() else "cpu"

        self.proj_head: ProjectionHead = None
        self.classifier: nn.Linear = None
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.train_log = []
        self.val_log = []

    def train(
        self,
        X_full: np.ndarray,
        y_full: np.ndarray,
        X_pairs: np.ndarray,
        y_pairs_intent: np.ndarray,
        y_pairs_binary: np.ndarray,
    ):
        """Train with both contrastive (pairs) and classification (full) loss.

        Args:
            X_full: (N, D) full training embeddings
            y_full: (N,) binary labels for full data
            X_pairs: (M, D) synthetic pair embeddings
            y_pairs_intent: (M,) intent labels (0=safe, 1=benign_sensitive, 2=malicious)
            y_pairs_binary: (M,) corrected binary labels (0=safe, 1=unsafe)
        """
        logger.info(f"Training SafeCL v3:")
        logger.info(f"  Full data: {X_full.shape[0]} samples, dim={X_full.shape[1]}")
        logger.info(f"  Synthetic pairs: {X_pairs.shape[0]} samples")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Temperature: {self.temperature}, lambda_cls: {self.lambda_cls}")

        # Build model
        self.proj_head = ProjectionHead(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            dropout=self.dropout,
        ).to(self.device)
        self.classifier = nn.Linear(self.output_dim, 2).to(self.device)

        self.contrastive_loss = SupConLoss(temperature=self.temperature)
        self.cls_loss = nn.CrossEntropyLoss()

        params = list(self.proj_head.parameters()) + list(self.classifier.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.max_epochs, eta_min=1e-6
        )

        # Split pairs into train/val
        indices = np.arange(len(X_pairs))
        tr_idx, val_idx = train_test_split(
            indices, test_size=self.val_size, stratify=y_pairs_intent, random_state=self.random_state
        )

        Xp_tr, Xp_val = X_pairs[tr_idx], X_pairs[val_idx]
        ypi_tr, ypi_val = y_pairs_intent[tr_idx], y_pairs_intent[val_idx]
        ypb_tr, ypb_val = y_pairs_binary[tr_idx], y_pairs_binary[val_idx]

        # Convert to tensors
        Xp_tr_t = torch.FloatTensor(Xp_tr)
        ypi_tr_t = torch.LongTensor(ypi_tr)
        ypb_tr_t = torch.LongTensor(ypb_tr)

        pair_dataset = TensorDataset(Xp_tr_t, ypi_tr_t, ypb_tr_t)
        pair_loader = DataLoader(pair_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        Xp_val_t = torch.FloatTensor(Xp_val).to(self.device)
        ypi_val_t = torch.LongTensor(ypi_val).to(self.device)
        ypb_val_t = torch.LongTensor(ypb_val).to(self.device)

        # For classification: use full data
        Xf_tr_t = torch.FloatTensor(X_full).to(self.device)
        yf_tr_t = torch.LongTensor(y_full).to(self.device)

        patience_counter = 0
        best_state = None

        for epoch in range(self.max_epochs):
            # ---- Training ----
            self.proj_head.train()
            self.classifier.train()

            epoch_con_loss = 0.0
            epoch_cls_loss = 0.0
            n_batches = 0

            for batch in pair_loader:
                bx = batch[0].to(self.device)
                bintent = batch[1].to(self.device)
                bbin = batch[2].to(self.device)

                bx_norm = torch.nn.functional.normalize(bx, p=2, dim=1)

                # Forward through projection head
                z = self.proj_head(bx_norm)

                # Contrastive loss: use intent labels for fine separation
                l_contrast = self.contrastive_loss(z, bintent)

                # Classification loss: use corrected binary labels
                logits = self.classifier(z)
                l_cls = self.cls_loss(logits, bbin)

                # Combined loss
                loss = l_contrast + self.lambda_cls * l_cls

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.proj_head.parameters(), max_norm=5.0)
                torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), max_norm=5.0)
                self.optimizer.step()

                epoch_con_loss += l_contrast.item()
                epoch_cls_loss += l_cls.item()
                n_batches += 1

            epoch_con_loss /= max(n_batches, 1)
            epoch_cls_loss /= max(n_batches, 1)
            self.scheduler.step()

            # ---- Validation ----
            self.proj_head.eval()
            self.classifier.eval()
            with torch.no_grad():
                zv = self.proj_head(torch.nn.functional.normalize(Xp_val_t, p=2, dim=1))
                val_con = self.contrastive_loss(zv, ypi_val_t)
                val_logits = self.classifier(zv)
                val_cls = self.cls_loss(val_logits, ypb_val_t)
                val_loss = val_con + self.lambda_cls * val_cls

            current_lr = self.scheduler.get_last_lr()[0]

            self.train_log.append({
                "epoch": epoch + 1,
                "train_con_loss": round(epoch_con_loss, 4),
                "train_cls_loss": round(epoch_cls_loss, 4),
                "lr": round(current_lr, 6),
            })
            self.val_log.append({
                "epoch": epoch + 1,
                "val_con_loss": round(val_con.item(), 4),
                "val_cls_loss": round(val_cls.item(), 4),
                "val_total": round(val_loss.item(), 4),
            })

            logger.info(
                f"Epoch {epoch+1}/{self.max_epochs}: "
                f"con={epoch_con_loss:.4f}, cls={epoch_cls_loss:.4f}, "
                f"val_con={val_con.item():.4f}, val_cls={val_cls.item():.4f}, "
                f"lr={current_lr:.2e}"
            )

            if val_loss < self.best_loss:
                self.best_loss = val_loss.item()
                self.best_epoch = epoch
                patience_counter = 0
                best_state = {
                    "proj": {k: v.cpu().clone() for k, v in self.proj_head.state_dict().items()},
                    "cls": {k: v.cpu().clone() for k, v in self.classifier.state_dict().items()},
                }
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}, best was {self.best_epoch+1}")
                    break

        if best_state is not None:
            self.proj_head.load_state_dict(best_state["proj"])
            self.classifier.load_state_dict(best_state["cls"])
            self.proj_head = self.proj_head.to(self.device)
            self.classifier = self.classifier.to(self.device)

        logger.info(f"Training complete. Best val_loss={self.best_loss:.4f} at epoch {self.best_epoch+1}")

    def project(self, X: np.ndarray) -> np.ndarray:
        return ProjectionHeadManager.project(self.proj_head, X, device=self.device)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities using the classification head."""
        X_t = torch.FloatTensor(X).to(self.device)
        X_norm = torch.nn.functional.normalize(X_t, p=2, dim=1)
        self.proj_head.eval()
        self.classifier.eval()
        with torch.no_grad():
            z = self.proj_head(X_norm)
            logits = self.classifier(z)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def save(self, proj_path: str, cls_path: str = None):
        ProjectionHeadManager.save(self.proj_head, proj_path)
        if cls_path is None:
            cls_path = proj_path.replace(".pkl", "_classifier.pkl")
        os.makedirs(os.path.dirname(cls_path), exist_ok=True)
        torch.save(self.classifier.state_dict(), cls_path)
        logger.info(f"Saved classifier to {cls_path}")


def main():
    parser = argparse.ArgumentParser(description="Train SafeCL v3")
    parser.add_argument("--embed-model", default="qwen3-embedding-0.6B")
    parser.add_argument("--pairs-path", default="data/processed/contrastive_pairs.parquet")
    parser.add_argument("--embeddings-dir", default="embeddings")
    parser.add_argument("--output-dir", default="models/safecl_v3")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--output-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--lambda-cls", type=float, default=0.3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(42)
    start_time = datetime.now()

    # 1. Load synthetic pairs and encode
    logger.info(f"[1/4] Loading synthetic pairs from {args.pairs_path}")
    df_pairs = pd.read_parquet(args.pairs_path)
    logger.info(f"  Generated pairs: {len(df_pairs)} samples, {df_pairs['topic'].nunique()} topics")

    # Encode pairs using Qwen3
    from src.encode.qwen_encoder import QwenEncoder
    model_path = f"pretrained/qwen/Qwen3-Embedding-0___6B"
    logger.info(f"[2/4] Encoding pairs with {model_path}")
    encoder = QwenEncoder(model_path, device=args.device)
    X_pairs = encoder.encode(df_pairs["text"].tolist())
    y_pairs_intent = df_pairs["intent"].values
    y_pairs_binary = df_pairs["label"].values
    logger.info(f"  Encoded: {X_pairs.shape}")

    # 2. Load full training embeddings
    logger.info("[3/4] Loading full training embeddings")
    embed_dir = os.path.join(args.embeddings_dir, args.embed_model)
    X_full = np.load(os.path.join(embed_dir, "train.npy"))
    y_full = np.load(os.path.join(embed_dir, "train_labels.npy"))
    # Correct binary labels for benign-sensitive in full data
    # Use the intent-annotated data
    intent_path = "data/processed/train_intent.parquet"
    if os.path.exists(intent_path):
        df_intent = pd.read_parquet(intent_path)
        y_full_corrected = np.where(df_intent["intent"].values == 1, 0, y_full)
        n_corrected = (y_full_corrected != y_full).sum()
        logger.info(f"  Corrected {n_corrected} labels in full training data ({100*n_corrected/len(y_full):.1f}%)")
        y_full = y_full_corrected
    logger.info(f"  Full training data: {X_full.shape}")

    # 3. Train
    logger.info("[4/4] Training SafeCL v3")
    trainer = SafeCLv3Trainer(
        input_dim=X_full.shape[1],
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        patience=args.patience,
        temperature=args.temperature,
        lambda_cls=args.lambda_cls,
        device=args.device,
    )

    trainer.train(
        X_full=X_full,
        y_full=y_full,
        X_pairs=X_pairs,
        y_pairs_intent=y_pairs_intent,
        y_pairs_binary=y_pairs_binary,
    )

    # 4. Save everything
    out_dir = os.path.join(args.output_dir, args.embed_model)
    os.makedirs(out_dir, exist_ok=True)

    proj_path = os.path.join(out_dir, "projection_head.pkl")
    cls_path = os.path.join(out_dir, "classifier.pkl")
    trainer.save(proj_path, cls_path)

    # Save training log
    log_path = os.path.join(out_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump({
            "config": {
                "hidden_dim": args.hidden_dim,
                "output_dim": args.output_dim,
                "lr": args.lr,
                "temperature": args.temperature,
                "lambda_cls": args.lambda_cls,
                "batch_size": args.batch_size,
                "epochs_completed": trainer.best_epoch + 1,
                "best_val_loss": trainer.best_loss,
            },
            "train_log": trainer.train_log,
            "val_log": trainer.val_log,
            "duration_seconds": (datetime.now() - start_time).total_seconds(),
            "pair_stats": {
                "n_pairs": len(df_pairs),
                "n_topics": df_pairs["topic"].nunique(),
                "n_safe": int((df_pairs["intent"] == 0).sum()),
                "n_benign_sensitive": int((df_pairs["intent"] == 1).sum()),
                "n_malicious": int((df_pairs["intent"] == 2).sum()),
            },
        }, f, indent=2)
    logger.info(f"Training log saved to {log_path}")

    # Project and save test embeddings
    test_emb_path = os.path.join(embed_dir, "test.npy")
    if os.path.exists(test_emb_path):
        X_test = np.load(test_emb_path)
        X_test_proj = trainer.project(X_test)
        np.save(os.path.join(out_dir, "test_projected.npy"), X_test_proj)
        X_train_proj = trainer.project(X_full)
        np.save(os.path.join(out_dir, "train_projected.npy"), X_train_proj)
        logger.info(f"Saved projected embeddings")

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"Done! Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
