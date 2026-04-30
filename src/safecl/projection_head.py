"""Learnable projection head for intent-aware safety representation.

Maps frozen Qwen3 embeddings (1024d/4096d) to a compact intent-aware space
where safe, benign-sensitive, and malicious samples are structurally separated.
"""

import logging
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ProjectionHead(nn.Module):
    """MLP projection head: Linear → BN → ReLU → Dropout → Linear → L2 Norm.

    Maps frozen embeddings to a lower-dimensional space optimized by
    hierarchical contrastive loss.
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

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass, returns L2-normalized embeddings."""
        z = self.net(x)
        return F.normalize(z, p=2, dim=1)

    def get_embedding_dim(self) -> int:
        return self.output_dim


class ProjectionHeadManager:
    """Utility class for saving/loading projection heads."""

    @staticmethod
    def save(model: ProjectionHead, path: str):
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
    def load(path: str, device: str = "cuda") -> ProjectionHead:
        with open(path, "rb") as f:
            save_dict = pickle.load(f)

        model = ProjectionHead(
            input_dim=save_dict["input_dim"],
            output_dim=save_dict["output_dim"],
        )
        model.load_state_dict(save_dict["state_dict"])
        model = model.to(device)
        model.eval()
        return model

    @staticmethod
    def project(
        model: ProjectionHead,
        embeddings: np.ndarray,
        batch_size: int = 1024,
        device: str = "cuda",
    ) -> np.ndarray:
        """Project embeddings through the trained head.

        Args:
            model: Trained ProjectionHead
            embeddings: (N, D) numpy array of frozen embeddings
            batch_size: Batch size for projection
            device: Device to run on

        Returns:
            (N, output_dim) projected embeddings
        """
        model = model.to(device)
        model.eval()

        projected = []
        with torch.no_grad():
            for i in range(0, len(embeddings), batch_size):
                batch = torch.FloatTensor(embeddings[i : i + batch_size]).to(device)
                z = model(batch)
                projected.append(z.cpu().numpy())

        return np.vstack(projected)
