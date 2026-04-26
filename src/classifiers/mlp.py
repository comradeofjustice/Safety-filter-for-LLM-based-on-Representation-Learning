"""MLP classifier with PyTorch."""

import logging
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class MLPModel(nn.Module):
    """MLP: Linear(D,1024) → BN → ReLU → Dropout(0.3) → Linear(1024,256) → BN → ReLU → Dropout(0.2) → Linear(256,64) → ReLU → Linear(64,2)"""

    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.network(x)


class MLPClassifier:
    """MLP classifier with early stopping."""

    def __init__(
        self,
        lr=5e-5,
        weight_decay=1e-3,
        batch_size=256,
        max_epoch=150,
        patience=15,
        random_state=42,
    ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.patience = patience
        self.random_state = random_state
        self.model = None
        self.input_dim = None
        self.trained = False

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the MLP with validation split and early stopping."""
        self.input_dim = X_train.shape[1]
        logger.info(f"Training MLP on {X_train.shape[0]} samples, dim={self.input_dim}...")

        # Split into train/val (90/10 stratified)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.1, stratify=y_train, random_state=self.random_state
        )

        # Convert to tensors
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_tr_t = torch.FloatTensor(X_tr).to(device)
        y_tr_t = torch.LongTensor(y_train if len(y_train) == len(X_tr) else y_tr).to(device)
        X_val_t = torch.FloatTensor(X_val).to(device)
        y_val_t = torch.LongTensor(y_val).to(device)

        # Use actual split labels
        y_tr_t = torch.LongTensor(y_tr).to(device)

        train_loader = DataLoader(
            TensorDataset(X_tr_t, y_tr_t),
            batch_size=self.batch_size,
            shuffle=True,
        )

        # Initialize model
        self.model = MLPModel(self.input_dim).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epoch, eta_min=1e-6
        )

        # Training loop with early stopping
        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0
        best_state = None

        for epoch in range(self.max_epoch):
            self.model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            scheduler.step()

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t).item()

            current_lr = scheduler.get_last_lr()[0]
            logger.info(
                f"Epoch {epoch+1}/{self.max_epoch}: train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, lr={current_lr:.2e}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}, best epoch was {best_epoch+1}")
                    break

        # Load best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model = self.model.to(device)

        self.trained = True
        logger.info(f"MLP training complete (best epoch: {best_epoch+1})")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels."""
        device = next(self.model.parameters()).device
        X_t = torch.FloatTensor(X).to(device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_t)
            _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        device = next(self.model.parameters()).device
        X_t = torch.FloatTensor(X).to(device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_t)
            probs = torch.softmax(outputs, dim=1)
        return probs.cpu().numpy()

    def save(self, path: str):
        """Save model to file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_dict = {
            "state_dict": {k: v.cpu() for k, v in self.model.state_dict().items()},
            "input_dim": self.input_dim,
        }
        with open(path, "wb") as f:
            pickle.dump(save_dict, f)
        logger.info(f"Saved MLP to {path}")

    @classmethod
    def load(cls, path: str) -> "MLPClassifier":
        """Load model from file."""
        with open(path, "rb") as f:
            save_dict = pickle.load(f)

        clf = cls()
        clf.input_dim = save_dict["input_dim"]
        clf.model = MLPModel(clf.input_dim)
        clf.model.load_state_dict(save_dict["state_dict"])
        clf.trained = True
        return clf
