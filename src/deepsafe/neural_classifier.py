"""DeepSafe Neural Classifier: Learnable classifier for projected safety embeddings.

Architecture: 3-layer residual MLP with temperature scaling.
  Input (output_dim) -> Linear -> LN -> GELU -> Dropout
    -> Linear -> LN -> GELU -> Dropout (with residual)
    -> Linear -> 2-class output
    -> Temperature scaling for calibration

Key features:
  - Residual connections for gradient flow
  - Temperature scaling for probability calibration
  - Class-weighted loss for imbalance
  - Mixup augmentation on embeddings
  - Configurable depth and width
"""

import logging
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

logger = logging.getLogger(__name__)


class NeuralClassifier(nn.Module):
    """Residual neural classifier for projected safety embeddings.

    Args:
        input_dim: Dimension of projected embeddings (typically 256)
        hidden_dims: List of hidden layer dimensions (e.g. [128, 64])
        dropout: Dropout rate
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dims: list = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        # Input normalization
        self.input_ln = nn.LayerNorm(input_dim)

        # Build layers
        layers = []
        in_dim = input_dim
        for i, hd in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, hd))
            layers.append(nn.LayerNorm(hd))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            # Residual adapter
            if i > 0:
                layers.append(ResidualBlock(hd, dropout))
            in_dim = hd
        self.backbone = nn.Sequential(*layers)

        # Classifier head
        self.head = nn.Linear(hidden_dims[-1], 2)

        # Temperature scaling for calibration
        self.log_temperature = nn.Parameter(torch.ones(1))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @property
    def temperature(self):
        return self.log_temperature.exp()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_ln(x)
        features = self.backbone(x)
        logits = self.head(features)
        return logits / self.temperature

    def predict_proba(self, x: np.ndarray, device: str = "cpu") -> np.ndarray:
        """Return probability scores for class 1 (unsafe)."""
        self.eval()
        with torch.no_grad():
            x_t = torch.FloatTensor(x).to(device)
            logits = self.forward(x_t)
            probs = F.softmax(logits, dim=-1)
            return probs[:, 1].cpu().numpy()

    def predict(self, x: np.ndarray, device: str = "cpu") -> np.ndarray:
        proba = self.predict_proba(x, device=device)
        return (proba > 0.5).astype(int)


class ResidualBlock(nn.Module):
    """Simple residual block with LayerNorm + GELU + Linear + Dropout."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.dropout(self.fc(self.act(self.ln(x))))


class NeuralClassifierTrainer:
    """Trainer for NeuralClassifier with mixup and calibration."""

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dims: list = None,
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 512,
        max_epochs: int = 50,
        patience: int = 10,
        mixup_alpha: float = 0.2,
        device: str = "cuda",
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [128, 64]
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.mixup_alpha = mixup_alpha
        self.device = device if torch.cuda.is_available() else "cpu"

        self.model: NeuralClassifier = None
        self.best_acc = 0.0
        self.best_state = None

    def _mixup_batch(self, x, y):
        """Apply mixup augmentation."""
        if self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        else:
            lam = 1.0
        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def _mixup_loss(self, logits, y_a, y_b, lam):
        """Cross-entropy loss for mixup."""
        log_probs = F.log_softmax(logits, dim=-1)
        loss_a = F.nll_loss(log_probs, y_a, reduction='none')
        loss_b = F.nll_loss(log_probs, y_b, reduction='none')
        return (lam * loss_a + (1 - lam) * loss_b).mean()

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
    ):
        logger.info(f"Training NeuralClassifier: {X.shape[0]} samples, dim={X.shape[1]}")

        # Split if no validation set provided
        if X_val is None:
            X, X_val, y, y_val = train_test_split(
                X, y, test_size=0.15, stratify=y, random_state=42
            )

        # Compute class weights
        unique, counts = np.unique(y, return_counts=True)
        n_samples = len(y)
        class_weights = torch.FloatTensor([n_samples / c for c in counts]).to(self.device)

        # Build model
        self.model = NeuralClassifier(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        ).to(self.device)

        # Optimizer & scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs, eta_min=1e-6
        )

        # DataLoader
        train_ds = TensorDataset(
            torch.FloatTensor(X), torch.LongTensor(y)
        )
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True, drop_last=False
        )

        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.LongTensor(y_val).to(self.device)

        patience_counter = 0

        for epoch in range(self.max_epochs):
            # Training
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()

                # Mixup
                if self.mixup_alpha > 0 and len(batch_x) > 1:
                    mixed_x, y_a, y_b, lam = self._mixup_batch(batch_x, batch_y)
                    logits = self.model(mixed_x)
                    loss = self._mixup_loss(logits, y_a, y_b, lam)
                else:
                    logits = self.model(batch_x)
                    loss = F.cross_entropy(logits, batch_y, weight=class_weights)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            epoch_loss /= max(n_batches, 1)
            scheduler.step()

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(X_val_t)
                val_probs = F.softmax(val_logits, dim=-1)
                val_preds = val_probs.argmax(dim=-1)
                val_acc = (val_preds == y_val_t).float().mean().item()
                val_score = val_probs[:, 1].cpu().numpy()
                try:
                    val_auc = roc_auc_score(y_val, val_score)
                except ValueError:
                    val_auc = 0.5

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"  Epoch {epoch+1}/{self.max_epochs}: "
                    f"loss={epoch_loss:.4f}, val_acc={val_acc:.4f}, val_auc={val_auc:.4f}"
                )

            if val_acc > self.best_acc:
                self.best_acc = val_acc
                patience_counter = 0
                self.best_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f"  Early stopping at epoch {epoch+1}, best val_acc={self.best_acc:.4f}")
                    break

        # Restore best
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
            self.model = self.model.to(self.device)

        # Final validation
        self.model.eval()
        with torch.no_grad():
            final_logits = self.model(X_val_t)
            final_probs = F.softmax(final_logits, dim=-1)
            final_preds = final_probs.argmax(dim=-1).cpu().numpy()
            final_score = final_probs[:, 1].cpu().numpy()

        test_acc = accuracy_score(y_val, final_preds)
        test_f1 = f1_score(y_val, final_preds)
        try:
            test_auc = roc_auc_score(y_val, final_score)
        except ValueError:
            test_auc = 0.5

        logger.info(f"  Final: acc={test_acc:.4f}, f1={test_f1:.4f}, auc={test_auc:.4f}")

        return {
            "accuracy": test_acc,
            "f1_macro": test_f1,
            "roc_auc": test_auc,
            "best_val_acc": self.best_acc,
        }

    def save(self, path: str):
        """Save classifier with metadata."""
        if self.model is None:
            raise RuntimeError("No trained model to save")
        save_dict = {
            "state_dict": {k: v.cpu() for k, v in self.model.state_dict().items()},
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
        }
        with open(path, "wb") as f:
            pickle.dump(save_dict, f)
        logger.info(f"Classifier saved to {path}")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> NeuralClassifier:
        with open(path, "rb") as f:
            save_dict = pickle.load(f)
        model = NeuralClassifier(
            input_dim=save_dict["input_dim"],
            hidden_dims=save_dict.get("hidden_dims", [128, 64]),
            dropout=save_dict.get("dropout", 0.2),
        )
        model.load_state_dict(save_dict["state_dict"])
        model = model.to(device)
        model.eval()
        return model
