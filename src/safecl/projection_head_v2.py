"""Improved projection head with residual connections for V7.

Key improvements over V1:
  1. Residual connection: Linear→LN→ReLU→Dropout→Linear with residual
  2. LayerNorm instead of BatchNorm (better for contrastive learning)
  3. Deeper architecture: 1024→1024→512→256
"""

import logging
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ResidualProjectionHead(nn.Module):
    """Residual projection head with LayerNorm for contrastive learning.

    Architecture:
      input → Linear(1024→1024) → LN → ReLU → Dropout
                ↓ (+ residual if dim matches)
              Linear(1024→512) → LN → ReLU → Dropout
                ↓
              Linear(512→256) → L2 Norm
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        output_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # First block with residual
        self.block1 = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Second block
        self.block2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1 with residual
        out = self.block1(x)
        out = out + x  # residual (same dim)

        # Block 2
        out = self.block2(out)

        # Output
        z = self.out_proj(out)
        return F.normalize(z, p=2, dim=1)


class ProjectionHeadManager:
    """Utility class for saving/loading projection heads."""

    @staticmethod
    def save(model: nn.Module, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_dict = {
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "input_dim": model.input_dim,
            "output_dim": model.output_dim,
        }
        with open(path, "wb") as f:
            pickle.dump(save_dict, f)
        logger.info(f"Saved ProjectionHead to {path}")

    @staticmethod
    def load(path: str, device: str = "cuda") -> nn.Module:
        with open(path, "rb") as f:
            save_dict = pickle.load(f)

        model = ResidualProjectionHead(
            input_dim=save_dict["input_dim"],
            output_dim=save_dict["output_dim"],
        )
        model.load_state_dict(save_dict["state_dict"])
        model = model.to(device)
        model.eval()
        return model

    @staticmethod
    def project(model: nn.Module, embeddings: np.ndarray, batch_size: int = 1024, device: str = "cuda") -> np.ndarray:
        model = model.to(device)
        model.eval()
        projected = []
        with torch.no_grad():
            for i in range(0, len(embeddings), batch_size):
                batch = torch.FloatTensor(embeddings[i : i + batch_size]).to(device)
                z = model(batch)
                projected.append(z.cpu().numpy())
        return np.vstack(projected)
