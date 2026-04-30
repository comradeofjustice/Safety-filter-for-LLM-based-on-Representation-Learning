"""Train SafeCL v9: Train on FULL data with explicit counterfactual pair repulsion.

Strategy: Use V1's proven full-data training pipeline but add an explicit
pair repulsion loss on synthetic counterfactual pairs. This keeps the
classification ability of full-data training while directly pushing apart
malicious/benign-sensitive pairs that share keywords.

Loss: L = L_hierarchical(full_data) + gamma * L_pair_repulsion(synthetic_pairs)

Where L_pair_repulsion = max(0, margin - distance(z_mal, z_ben))^2
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
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.safecl.projection_head import ProjectionHead, ProjectionHeadManager
from src.safecl.losses import HierarchicalContrastiveLoss, SupConLoss
from src.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)




class PairRepulsionLoss(nn.Module):
    """Maximize distance between paired malicious/benign_sensitive samples."""

    def __init__(self, margin: float = 1.2):
        super().__init__()
        self.margin = margin

    def forward(self, z: torch.Tensor, pair_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (N, dim) L2-normalized embeddings
            pair_indices: (N,) indices where consecutive malicious/benign
                          from same topic have same index
        Returns:
            Scalar repulsion loss
        """
        device = z.device
        batch_size = z.shape[0]

        if batch_size < 2:
            return torch.tensor(0.0, device=device)

        # Compute pairwise distances
        loss = 0.0
        n_pairs = 0

        # Find unique pair indices in this batch
        unique_pairs = torch.unique(pair_indices)

        for pid in unique_pairs:
            mask = pair_indices == pid
            indices = torch.where(mask)[0]

            if len(indices) >= 2:
                # Push all pairs with same pair_id apart
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        dist = torch.norm(z[indices[i]] - z[indices[j]], p=2)
                        loss += torch.relu(self.margin - dist) ** 2
                        n_pairs += 1

        if n_pairs > 0:
            return loss / n_pairs
        return torch.tensor(0.0, device=device)


class SafeCLv9Trainer:
    """Train projection head with hierarchical loss on full data + repulsion on pairs."""

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        output_dim: int = 256,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        max_epochs: int = 50,
        patience: int = 12,
        temperature: float = 0.07,
        alpha: float = 1.5,       # Higher intent weight for better separation
        beta: float = 0.3,
        gamma: float = 0.5,        # Weight for pair repulsion loss
        repulsion_margin: float = 1.2,
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
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.repulsion_margin = repulsion_margin
        self.val_size = val_size
        self.random_state = random_state
        self.device = device

        self.model: ProjectionHead = None
        self.criterion: HierarchicalContrastiveLoss = None
        self.repulsion_loss: PairRepulsionLoss = None
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.train_log = []
        self.val_log = []

    def train(
        self,
        train_embeddings: np.ndarray,
        train_intent_labels: np.ndarray,
        train_binary_labels: np.ndarray,
        X_pairs: np.ndarray = None,
        y_pairs_intent: np.ndarray = None,
        pair_indices: np.ndarray = None,
        finegrained_labels: np.ndarray = None,
    ):
        logger.info(f"Training SafeCL v9: {train_embeddings.shape[0]} samples, alpha={self.alpha}, gamma={self.gamma}")
        if X_pairs is not None:
            logger.info(f"  Pair repulsion: {X_pairs.shape[0]} samples")

        # Split full data into train/val
        indices = np.arange(len(train_embeddings))
        tr_idx, val_idx = train_test_split(
            indices, test_size=self.val_size, stratify=train_intent_labels,
            random_state=self.random_state
        )

        X_tr = train_embeddings[tr_idx]
        X_val = train_embeddings[val_idx]
        y_tr_intent = train_intent_labels[tr_idx]
        y_val_intent = train_intent_labels[val_idx]
        y_tr_binary = train_binary_labels[tr_idx]
        y_val_binary = train_binary_labels[val_idx]

        # Create dataloaders
        X_tr_t = torch.FloatTensor(X_tr)
        y_tr_intent_t = torch.LongTensor(y_tr_intent)
        y_tr_binary_t = torch.LongTensor(y_tr_binary)

        train_dataset = TensorDataset(X_tr_t, y_tr_intent_t, y_tr_binary_t)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_intent_t = torch.LongTensor(y_val_intent).to(self.device)
        y_val_binary_t = torch.LongTensor(y_val_binary).to(self.device)

        # Prepare pair data
        if X_pairs is not None:
            # Split pairs into train/val
            pair_indices_arr = np.arange(len(X_pairs))
            p_tr_idx, p_val_idx = train_test_split(
                pair_indices_arr, test_size=self.val_size,
                stratify=y_pairs_intent, random_state=self.random_state
            )
            Xp_tr = torch.FloatTensor(X_pairs[p_tr_idx])
            ypi_tr = torch.LongTensor(y_pairs_intent[p_tr_idx])
            pid_tr = torch.LongTensor(pair_indices[p_tr_idx])

            pair_dataset = TensorDataset(Xp_tr, ypi_tr, pid_tr)
            pair_loader = DataLoader(pair_dataset, batch_size=min(64, len(p_tr_idx)),
                                     shuffle=True, drop_last=False)

            Xp_val_t = torch.FloatTensor(X_pairs[p_val_idx]).to(self.device)
            ypi_val_t = torch.LongTensor(y_pairs_intent[p_val_idx]).to(self.device)
            pid_val_t = torch.LongTensor(pair_indices[p_val_idx]).to(self.device)
        else:
            pair_loader = None

        # Build model
        self.model = ProjectionHead(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            dropout=self.dropout,
        ).to(self.device)

        self.criterion = HierarchicalContrastiveLoss(
            temperature=self.temperature, alpha=self.alpha, beta=self.beta
        )
        self.repulsion_loss = PairRepulsionLoss(margin=self.repulsion_margin)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=1e-6)

        patience_counter = 0
        best_state = None

        # Prepare pair loader iterator for interleaved training
        pair_iter = iter(pair_loader) if pair_loader is not None else None

        for epoch in range(self.max_epochs):
            self.model.train()
            epoch_binary = 0.0
            epoch_intent = 0.0
            epoch_rep = 0.0
            epoch_total = 0.0
            n_batches = 0

            for batch in train_loader:
                bx = batch[0].to(self.device)
                bintent = batch[1].to(self.device)
                bbin = batch[2].to(self.device)

                bx = F.normalize(bx, p=2, dim=1)
                z = self.model(bx)

                # Hierarchical contrastive loss on full data
                losses = self.criterion(z, bbin, bintent, None)
                total_loss = losses["total"]
                l_rep_val = 0.0

                # Add pair repulsion loss (interleaved, single backward)
                if pair_iter is not None:
                    try:
                        pbatch = next(pair_iter)
                    except StopIteration:
                        pair_iter = iter(pair_loader)
                        pbatch = next(pair_iter)

                    px = pbatch[0].to(self.device)
                    ppid = pbatch[2].to(self.device)
                    px = F.normalize(px, p=2, dim=1)
                    pz = self.model(px)
                    l_rep = self.repulsion_loss(pz, ppid)
                    total_loss = total_loss + self.gamma * l_rep
                    l_rep_val = l_rep.item()

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()

                epoch_binary += losses["binary"].item()
                epoch_intent += losses["intent"].item()
                epoch_rep += l_rep_val
                epoch_total += total_loss.item()
                n_batches += 1

            epoch_binary /= max(n_batches, 1)
            epoch_intent /= max(n_batches, 1)
            epoch_total /= max(n_batches, 1)
            epoch_rep /= max(n_batches, 1)

            scheduler.step()

            # Validation
            self.model.eval()
            with torch.no_grad():
                zv = self.model(F.normalize(X_val_t, p=2, dim=1))
                val_losses = self.criterion(zv, y_val_binary_t, y_val_intent_t, None)
                val_loss = val_losses["total"]

                if X_pairs is not None:
                    pzv = self.model(F.normalize(Xp_val_t, p=2, dim=1))
                    val_rep = self.repulsion_loss(pzv, pid_val_t)
                else:
                    val_rep = torch.tensor(0.0)

            current_lr = scheduler.get_last_lr()[0]

            self.train_log.append({
                "epoch": epoch + 1,
                "binary_loss": round(epoch_binary, 4),
                "intent_loss": round(epoch_intent, 4),
                "repulsion_loss": round(epoch_rep, 4),
                "total_loss": round(epoch_total, 4),
                "lr": round(current_lr, 6),
            })
            self.val_log.append({
                "epoch": epoch + 1,
                "val_total": round(val_loss.item(), 4),
                "val_repulsion": round(val_rep.item(), 4) if isinstance(val_rep, torch.Tensor) else 0,
            })

            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch+1}/{self.max_epochs}: "
                    f"bin={epoch_binary:.4f}, intent={epoch_intent:.4f}, "
                    f"rep={epoch_rep:.4f}, val={val_loss.item():.4f}"
                )

            if val_loss < self.best_loss:
                self.best_loss = val_loss.item()
                self.best_epoch = epoch + 1
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model = self.model.to(self.device)

        logger.info(f"Training complete. Best val_loss={self.best_loss:.4f} at epoch {self.best_epoch}")

    def project(self, X: np.ndarray) -> np.ndarray:
        return ProjectionHeadManager.project(self.model, X, device=self.device)

    def save(self, path: str):
        ProjectionHeadManager.save(self.model, path)


def main():
    parser = argparse.ArgumentParser(description="Train SafeCL v9")
    parser.add_argument("--embed-model", default="qwen3-embedding-0.6B")
    parser.add_argument("--embeddings-dir", default="embeddings")
    parser.add_argument("--output-dir", default="models/safecl_v9")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--output-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--alpha", type=float, default=1.5, help="Intent loss weight (higher = more separation)")
    parser.add_argument("--gamma", type=float, default=0.5, help="Pair repulsion weight")
    parser.add_argument("--repulsion-margin", type=float, default=1.2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(42)
    start_time = datetime.now()

    # 1. Load full training data
    logger.info("[1/3] Loading full training data")
    embed_dir = os.path.join(args.embeddings_dir, args.embed_model)
    X_train = np.load(os.path.join(embed_dir, "train.npy"))
    y_train = np.load(os.path.join(embed_dir, "train_labels.npy"))

    # Load intent labels
    intent_path = "data/processed/train_intent.parquet"
    df_intent = pd.read_parquet(intent_path)
    y_intent = df_intent["intent"].values

    # Correct binary labels
    from src.safecl.intent import IntentTaxonomy
    y_binary_corrected = np.array([IntentTaxonomy.to_binary(i) for i in y_intent])
    n_corrected = (y_binary_corrected != y_train).sum()
    logger.info(f"  Corrected {n_corrected} binary labels ({100*n_corrected/len(y_train):.1f}%)")

    # 2. Load or generate synthetic pairs
    logger.info("[2/3] Loading synthetic pairs")
    pairs_path = "data/processed/contrastive_pairs.parquet"
    df_pairs = pd.read_parquet(pairs_path)
    logger.info(f"  Loaded {len(df_pairs)} pairs from {pairs_path}")

    # Encode pairs
    embed_dir_local = os.path.join(args.embeddings_dir, args.embed_model)
    pairs_emb_path = os.path.join(embed_dir_local, "pairs.npy")
    if os.path.exists(pairs_emb_path):
        X_pairs = np.load(pairs_emb_path)
        logger.info(f"  Loaded cached pair embeddings: {X_pairs.shape}")
    else:
        from src.encode.qwen_encoder import QwenEncoder
        encoder = QwenEncoder("pretrained/qwen/Qwen3-Embedding-0___6B", device="cpu")
        X_pairs = encoder.encode(df_pairs["text"].tolist())
        np.save(pairs_emb_path, X_pairs)
        logger.info(f"  Encoded and cached: {X_pairs.shape}")

    y_pairs_intent = df_pairs["intent"].values.astype(np.int64)

    # Create pair indices
    pair_ids = df_pairs["pair_id"].values
    unique_pids = np.unique(pair_ids)
    pid_map = {pid: i for i, pid in enumerate(unique_pids)}
    pair_indices = np.array([pid_map[pid] for pid in pair_ids])

    # 3. Train
    logger.info("[3/3] Training SafeCL v9")
    trainer = SafeCLv9Trainer(
        input_dim=X_train.shape[1],
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        dropout=0.1,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        patience=args.patience,
        temperature=args.temperature,
        alpha=args.alpha,
        gamma=args.gamma,
        repulsion_margin=args.repulsion_margin,
        device=args.device,
    )

    trainer.train(
        train_embeddings=X_train,
        train_intent_labels=y_intent,
        train_binary_labels=y_binary_corrected,
        X_pairs=X_pairs,
        y_pairs_intent=y_pairs_intent,
        pair_indices=pair_indices,
    )

    # 4. Save
    out_dir = os.path.join(args.output_dir, args.embed_model)
    os.makedirs(out_dir, exist_ok=True)
    proj_path = os.path.join(out_dir, "projection_head.pkl")
    trainer.save(proj_path)

    # Save training log
    log_path = os.path.join(out_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump({
            "config": {
                "hidden_dim": args.hidden_dim,
                "output_dim": args.output_dim,
                "lr": args.lr,
                "temperature": args.temperature,
                "alpha": args.alpha,
                "gamma": args.gamma,
                "repulsion_margin": args.repulsion_margin,
                "epochs_completed": trainer.best_epoch,
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
        }, f, indent=2, default=str)
    logger.info(f"Training log saved to {log_path}")

    # Project and save embeddings
    test_emb_path = os.path.join(embed_dir, "test.npy")
    if os.path.exists(test_emb_path):
        X_test = np.load(test_emb_path)
        X_test_proj = trainer.project(X_test)
        np.save(os.path.join(out_dir, "test_projected.npy"), X_test_proj)
        X_train_proj = trainer.project(X_train)
        np.save(os.path.join(out_dir, "train_projected.npy"), X_train_proj)

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"Done! Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
