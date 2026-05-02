"""Robust projection head: simpler 2-layer MLP with residual connection + advanced losses.

This is a more robust alternative to the hyperbolic projection head.
Architecture: Input -> LayerNorm -> Linear(512) -> GELU -> Dropout -> Linear(256) -> L2Norm

Combined with the same advanced loss functions (hierarchical + OT + prototype + decorrelation).
"""

import logging
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class RobustProjectionHead(nn.Module):
    """Simple but effective projection head with residual connection.

    Maps frozen embeddings to a compact safety-aware representation space.
    Uses spectral normalization for stability.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        output_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        # Layer normalization on input
        self.input_norm = nn.LayerNorm(input_dim)

        # Main projection
        self.fc1 = nn.utils.spectral_norm(nn.Linear(input_dim, hidden_dim))
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # Residual adapter
        self.res_adapter = nn.Linear(input_dim, hidden_dim)

        # Output
        self.fc2 = nn.utils.spectral_norm(nn.Linear(hidden_dim, output_dim))
        self.bn2 = nn.BatchNorm1d(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input normalization
        x_norm = self.input_norm(x)

        # Main pathway
        h = self.fc1(x_norm)
        h = self.bn1(h)
        h = self.act(h)

        # Residual
        res = self.res_adapter(x_norm)
        h = h + res

        h = self.dropout(h)

        # Output
        z = self.fc2(h)
        z = self.bn2(z)

        return F.normalize(z, p=2, dim=1)


class RobustHeadManager:
    """Save/load/project utilities."""

    @staticmethod
    def save(model: RobustProjectionHead, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_dict = {
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "input_dim": model.input_dim,
            "output_dim": model.output_dim,
            "hidden_dim": model.hidden_dim,
        }
        with open(path, "wb") as f:
            pickle.dump(save_dict, f)

    @staticmethod
    def load(path: str, device: str = "cuda") -> RobustProjectionHead:
        with open(path, "rb") as f:
            save_dict = pickle.load(f)

        model = RobustProjectionHead(
            input_dim=save_dict["input_dim"],
            hidden_dim=save_dict.get("hidden_dim", 512),
            output_dim=save_dict["output_dim"],
        )
        model.load_state_dict(save_dict["state_dict"])
        model = model.to(device)
        model.eval()
        return model

    @staticmethod
    def project(model: RobustProjectionHead, embeddings: np.ndarray,
                batch_size: int = 1024, device: str = "cuda") -> np.ndarray:
        model = model.to(device)
        model.eval()
        projected = []
        with torch.no_grad():
            for i in range(0, len(embeddings), batch_size):
                batch = torch.FloatTensor(embeddings[i: i + batch_size]).to(device)
                z = model(batch)
                projected.append(z.cpu().numpy())
        return np.vstack(projected)
