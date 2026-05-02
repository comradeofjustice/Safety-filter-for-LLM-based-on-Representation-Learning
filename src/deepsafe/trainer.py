"""DeepSafe Trainer: trains advanced hyperbolic projection head.

Training strategy:
  - Load frozen embeddings + intent labels
  - Train DeepSafeProjectionHead with combined loss
  - AdamW + CosineAnnealingLR + early stopping
  - Counterfactual pairs interleaved during training
"""

import logging
import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split

from .projection_head import DeepSafeProjectionHead, DeepSafeProjectionHeadManager
from .losses import DeepSafeLoss

logger = logging.getLogger(__name__)


class DeepSafeTrainer:
    """Trainer for DeepSafe projection head.

    Trains an advanced projection head combining hyperbolic geometry,
    optimal transport, prototype learning, and spectral decorrelation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        output_dim: int = 256,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        max_epochs: int = 100,
        patience: int = 15,
        temperature: float = 0.07,
        alpha: float = 2.0,
        lambda_ot: float = 0.3,
        lambda_proto: float = 0.2,
        lambda_decorr: float = 0.005,
        gamma: float = 0.5,
        val_size: float = 0.1,
        random_state: int = 42,
        device: str = "cuda",
        use_amp: bool = False,
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
        self.lambda_ot = lambda_ot
        self.lambda_proto = lambda_proto
        self.lambda_decorr = lambda_decorr
        self.gamma = gamma
        self.val_size = val_size
        self.random_state = random_state
        self.device = device if torch.cuda.is_available() else "cpu"
        self.use_amp = use_amp

        self.model: DeepSafeProjectionHead = None
        self.criterion: DeepSafeLoss = None
        self.optimizer = None
        self.scheduler = None
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.train_history = []
        self.val_history = []

    def _build_model(self):
        self.model = DeepSafeProjectionHead(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            dropout=self.dropout,
        ).to(self.device)

        self.criterion = DeepSafeLoss(
            feature_dim=self.output_dim,
            temperature=self.temperature,
            alpha=self.alpha,
            lambda_ot=self.lambda_ot,
            lambda_proto=self.lambda_proto,
            lambda_decorr=self.lambda_decorr,
            gamma=self.gamma,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.max_epochs, eta_min=1e-6
        )

        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

    def train(
        self,
        train_embeddings: np.ndarray,
        train_intent_labels: np.ndarray,
        train_binary_labels: np.ndarray,
        cf_pairs: tuple = None,
    ):
        """Train DeepSafe projection head.

        Args:
            train_embeddings: (N, D) frozen Qwen3 embeddings
            train_intent_labels: (N,) 0=safe, 1=benign_sensitive, 2=malicious
            train_binary_labels: (N,) 0=safe, 1=unsafe
            cf_pairs: optional (emb_i, emb_j) tuple of counterfactual pairs
        """
        logger.info(
            f"Training DeepSafe: {train_embeddings.shape[0]} samples, "
            f"dim={train_embeddings.shape[1]}, device={self.device}"
        )

        # Split into train/val
        indices = np.arange(len(train_embeddings))
        tr_idx, val_idx = train_test_split(
            indices,
            test_size=self.val_size,
            stratify=train_intent_labels,
            random_state=self.random_state,
        )

        X_tr = train_embeddings[tr_idx]
        X_val = train_embeddings[val_idx]
        y_tr_intent = train_intent_labels[tr_idx]
        y_val_intent = train_intent_labels[val_idx]
        y_tr_binary = train_binary_labels[tr_idx]
        y_val_binary = train_binary_labels[val_idx]

        # Build dataloader
        train_dataset = TensorDataset(
            torch.FloatTensor(X_tr),
            torch.LongTensor(y_tr_intent),
            torch.LongTensor(y_tr_binary),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        # Validation tensors (pre-moved to GPU)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_intent_t = torch.LongTensor(y_val_intent).to(self.device)
        y_val_binary_t = torch.LongTensor(y_val_binary).to(self.device)

        # Counterfactual pairs
        cf_tensors = None
        if cf_pairs is not None:
            cf_emb_i, cf_emb_j = cf_pairs
            cf_tensors = (
                torch.FloatTensor(cf_emb_i).to(self.device),
                torch.FloatTensor(cf_emb_j).to(self.device),
            )

        # Build model
        self._build_model()

        patience_counter = 0
        best_state = None

        for epoch in range(self.max_epochs):
            # Training
            self.model.train()
            epoch_losses = {
                "total": 0, "hierarchical": 0, "binary": 0, "intent": 0,
                "sinkhorn_ot": 0, "prototype": 0, "decorrelation": 0, "counterfactual": 0,
            }
            n_batches = 0

            for batch in train_loader:
                batch_x = batch[0].to(self.device)
                batch_intent = batch[1].to(self.device)
                batch_binary = batch[2].to(self.device)

                self.optimizer.zero_grad()
                features = self.model(batch_x)
                loss, loss_components = self.criterion(
                    features, batch_binary, batch_intent,
                    return_components=True,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()

                for k in epoch_losses:
                    if k in loss_components:
                        epoch_losses[k] += loss_components[k]
                n_batches += 1

            # Interleave counterfactual pairs
            if cf_tensors is not None:
                self.model.train()
                cf_emb_i, cf_emb_j = cf_tensors
                # Process in sub-batches
                cf_bs = min(self.batch_size, len(cf_emb_i))
                for cf_start in range(0, len(cf_emb_i), cf_bs):
                    cf_end = min(cf_start + cf_bs, len(cf_emb_i))
                    z_i = self.model(cf_emb_i[cf_start:cf_end])
                    z_j = self.model(cf_emb_j[cf_start:cf_end])
                    cf_loss = self.criterion.counterfactual(z_i, z_j) * self.gamma
                    self.optimizer.zero_grad()
                    cf_loss.backward()
                    self.optimizer.step()

            # Average losses
            for k in epoch_losses:
                epoch_losses[k] /= max(n_batches, 1)

            self.scheduler.step()

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_features = self.model(X_val_t)
                val_loss, val_components = self.criterion(
                    val_features, y_val_binary_t, y_val_intent_t,
                    return_components=True,
                )

            current_lr = self.scheduler.get_last_lr()[0]
            self.train_history.append(epoch_losses)
            self.val_history.append(val_components)

            logger.info(
                f"Epoch {epoch+1}/{self.max_epochs}: "
                f"train={epoch_losses['total']:.4f}, val={val_components['total']:.4f}, "
                f"hier={epoch_losses.get('hierarchical', 0):.4f}, "
                f"ot={epoch_losses.get('sinkhorn_ot', 0):.4f}, "
                f"proto={epoch_losses.get('prototype', 0):.4f}, lr={current_lr:.2e}"
            )

            # Early stopping
            val_total = val_components["total"]
            if val_total < self.best_loss:
                self.best_loss = val_total
                self.best_epoch = epoch
                patience_counter = 0
                best_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(
                        f"Early stopping at epoch {epoch+1}, "
                        f"best was epoch {self.best_epoch+1}"
                    )
                    break

            # Free memory after each epoch
            if self.device == "cuda":
                torch.cuda.empty_cache()

        # Restore best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model = self.model.to(self.device)

        logger.info(
            f"Training complete. Best val_loss={self.best_loss:.4f} "
            f"at epoch {self.best_epoch+1}"
        )

    def project(self, embeddings: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        """Project embeddings through the trained projection head."""
        return DeepSafeProjectionHeadManager.project(
            self.model, embeddings, device=self.device, batch_size=batch_size
        )

    def save(self, path: str):
        """Save trained projection head."""
        DeepSafeProjectionHeadManager.save(self.model, path)

    @classmethod
    def load(cls, path: str, device: str = "cuda") -> "DeepSafeTrainer":
        """Load trained projection head."""
        model = DeepSafeProjectionHeadManager.load(path, device=device)
        trainer = cls(input_dim=model.input_dim, output_dim=model.output_dim)
        trainer.model = model
        return trainer
