"""SafeCL trainer: trains projection head with hierarchical contrastive loss."""

import logging
import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from .projection_head import ProjectionHead, ProjectionHeadManager
from .losses import HierarchicalContrastiveLoss

logger = logging.getLogger(__name__)


class SafeCLTrainer:
    """Trainer for the SafeCL projection head.

    Trains a projection head that maps frozen Qwen3 embeddings to an
    intent-aware safety representation space using hierarchical contrastive loss.

    Training strategy:
    - Load frozen embeddings + intent labels
    - Random projection head weights
    - Hierarchical contrastive loss at multiple label granularities
    - AdamW + CosineAnnealingLR + Early stopping
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
        alpha: float = 0.5,
        beta: float = 0.3,
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
        self.val_size = val_size
        self.random_state = random_state
        self.device = device if torch.cuda.is_available() else "cpu"

        self.model: ProjectionHead = None
        self.criterion: HierarchicalContrastiveLoss = None
        self.optimizer = None
        self.scheduler = None
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.train_history = []
        self.val_history = []

    def _build_model(self):
        self.model = ProjectionHead(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            dropout=self.dropout,
        ).to(self.device)

        self.criterion = HierarchicalContrastiveLoss(
            temperature=self.temperature,
            alpha=self.alpha,
            beta=self.beta,
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.max_epochs, eta_min=1e-6
        )

    def train(
        self,
        train_embeddings: np.ndarray,
        train_intent_labels: np.ndarray,
        train_binary_labels: np.ndarray,
        finegrained_labels: np.ndarray = None,
    ):
        """Train the projection head.

        Args:
            train_embeddings: (N, D) frozen Qwen3 embeddings
            train_intent_labels: (N,) 0=safe, 1=benign_sensitive, 2=malicious
            train_binary_labels: (N,) 0=safe, 1=unsafe
            finegrained_labels: (N,) optional fine-grained labels
        """
        logger.info(
            f"Training SafeCL: {train_embeddings.shape[0]} samples, "
            f"dim={train_embeddings.shape[1]}, device={self.device}"
        )

        # Split into train/val (stratified by intent)
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

        if finegrained_labels is not None:
            y_tr_fine = finegrained_labels[tr_idx]
            y_val_fine = finegrained_labels[val_idx]
        else:
            y_tr_fine = None
            y_val_fine = None

        # Convert to tensors and create dataloaders
        X_tr_t = torch.FloatTensor(X_tr)
        y_tr_intent_t = torch.LongTensor(y_tr_intent)
        y_tr_binary_t = torch.LongTensor(y_tr_binary)
        y_tr_fine_t = torch.LongTensor(y_tr_fine) if y_tr_fine is not None else None

        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_intent_t = torch.LongTensor(y_val_intent).to(self.device)
        y_val_binary_t = torch.LongTensor(y_val_binary).to(self.device)
        y_val_fine_t = torch.LongTensor(y_val_fine).to(self.device) if y_val_fine is not None else None

        tensors = [X_tr_t, y_tr_intent_t, y_tr_binary_t]
        self._has_finegrained = y_tr_fine_t is not None
        if self._has_finegrained:
            tensors.append(y_tr_fine_t)

        train_dataset = TensorDataset(*tensors)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        # Build model
        self._build_model()

        patience_counter = 0
        best_state = None

        for epoch in range(self.max_epochs):
            # Training
            self.model.train()
            epoch_losses = {"total": 0, "binary": 0, "intent": 0, "finegrained": 0}
            n_batches = 0

            for batch in train_loader:
                batch_x = batch[0].to(self.device)
                batch_intent = batch[1].to(self.device)
                batch_binary = batch[2].to(self.device)
                batch_fine = batch[3].to(self.device) if self._has_finegrained else None

                # Normalize input embeddings (already L2-normalized, but ensure)
                batch_x = torch.nn.functional.normalize(batch_x, p=2, dim=1)

                # Forward pass through projection head
                features = self.model(batch_x)

                # Compute hierarchical contrastive loss
                if batch_fine is not None and batch_fine.max() >= 0:
                    losses = self.criterion(
                        features, batch_binary, batch_intent, batch_fine
                    )
                else:
                    losses = self.criterion(
                        features, batch_binary, batch_intent, None
                    )

                self.optimizer.zero_grad()
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()

                for k in epoch_losses:
                    epoch_losses[k] += losses[k].item()
                n_batches += 1

            # Average losses
            for k in epoch_losses:
                epoch_losses[k] /= max(n_batches, 1)

            self.scheduler.step()

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_features = self.model(
                    torch.nn.functional.normalize(X_val_t, p=2, dim=1)
                )
                val_losses = self.criterion(
                    val_features, y_val_binary_t, y_val_intent_t, y_val_fine_t
                )

            current_lr = self.scheduler.get_last_lr()[0]

            self.train_history.append(epoch_losses)
            self.val_history.append({k: v.item() for k, v in val_losses.items()})

            logger.info(
                f"Epoch {epoch+1}/{self.max_epochs}: "
                f"train_loss={epoch_losses['total']:.4f} (b={epoch_losses['binary']:.4f}, "
                f"i={epoch_losses['intent']:.4f}, f={epoch_losses['finegrained']:.4f}), "
                f"val_loss={val_losses['total'].item():.4f}, lr={current_lr:.2e}"
            )

            # Early stopping
            val_total = val_losses["total"].item()
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
                        f"best epoch was {self.best_epoch+1}"
                    )
                    break

        # Restore best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model = self.model.to(self.device)

        logger.info(
            f"Training complete. Best val_loss={self.best_loss:.4f} "
            f"at epoch {self.best_epoch+1}"
        )

    def project(self, embeddings: np.ndarray) -> np.ndarray:
        """Project embeddings through the trained projection head."""
        return ProjectionHeadManager.project(
            self.model, embeddings, device=self.device
        )

    def save(self, path: str):
        """Save trained projection head."""
        ProjectionHeadManager.save(self.model, path)

    @classmethod
    def load(cls, path: str, device: str = "cuda") -> "SafeCLTrainer":
        """Load trained projection head (returns a lightweight wrapper)."""
        model = ProjectionHeadManager.load(path, device=device)
        trainer = cls(input_dim=model.input_dim, output_dim=model.output_dim)
        trainer.model = model
        return trainer
