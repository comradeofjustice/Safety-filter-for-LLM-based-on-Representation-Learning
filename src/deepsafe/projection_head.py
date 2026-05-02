"""Advanced projection head with hyperbolic layers and spectral decorrelation.

Architecture:
  1. Input L2 normalization
  2. Exponential map -> Poincaré ball (tangent -> manifold)
  3. Hyperbolic Linear (Möbius transform)
  4. Hyperbolic Activation (ReLU in tangent space)
  5. Log map -> Euclidean (manifold -> tangent)
  6. Residual MLP with LayerNorm
  7. L2 Normalize output

Key innovations over original MLP:
  - Hyperbolic geometry naturally encodes hierarchical safety taxonomy
  - Distance to origin captures "severity" level in the safety hierarchy
  - Residual connection preserves original embedding information
  - Spectral decorrelation promotes diverse feature usage
"""

import logging
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hyperbolic import (
    HyperbolicLinear,
    HyperbolicActivation,
    HyperbolicToEuclidean,
    exp_map,
    project_onto_ball,
)

logger = logging.getLogger(__name__)


class SpectralDecorrelation(nn.Module):
    """Barlow Twins-style cross-correlation matrix regularization.

    Penalizes off-diagonal elements of the feature correlation matrix,
    promoting diverse, non-redundant feature dimensions.
    """

    def __init__(self, lamb=0.005):
        super().__init__()
        self.lamb = lamb

    def forward(self, z):
        """Compute decorrelation loss.

        Args:
            z: (B, D) L2-normalized embeddings

        Returns:
            Scalar decorrelation loss
        """
        B, D = z.shape
        # Compute correlation matrix
        corr = torch.matmul(z.t(), z) / B

        # Penalize off-diagonal elements
        off_diag = corr - torch.diag(torch.diag(corr))
        loss = self.lamb * (off_diag ** 2).sum() / D

        return loss


class DeepSafeProjectionHead(nn.Module):
    """Advanced projection head combining hyperbolic geometry + residual MLP.

    Maps frozen embeddings (1024d/4096d) -> hyperbolic space -> compact Euclidean
    output (256d) with hierarchical structure preserved.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        output_dim: int = 256,
        dropout: float = 0.1,
        curvature: float = 1.0,
        hyperbolic_dim: int = 128,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.curvature = curvature
        self.hyperbolic_dim = hyperbolic_dim

        # 1. Project input to a lower-dim tangent space for hyperbolic mapping
        self.input_proj = nn.Linear(input_dim, hyperbolic_dim)
        self.input_ln = nn.LayerNorm(hyperbolic_dim)

        # 2. Hyperbolic layers (operate in Poincaré ball)
        self.hyp_linear = HyperbolicLinear(hyperbolic_dim, hyperbolic_dim, c=curvature)
        self.hyp_act = HyperbolicActivation(activation=nn.ReLU(), c=curvature)

        # 3. Bridge from hyperbolic to Euclidean
        self.hyp_to_euc = HyperbolicToEuclidean(hyperbolic_dim, hidden_dim, c=curvature)

        # 4. Residual MLP
        self.residual_proj = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # 5. Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
        )

        # Learnable temperature for contrastive learning
        self.log_temperature = nn.Parameter(torch.tensor(0.07).log())

    def forward(self, x: torch.Tensor, return_hidden: bool = False) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, input_dim) frozen embeddings
            return_hidden: if True, return pre-L2-norm features

        Returns:
            (B, output_dim) L2-normalized projected embeddings
        """
        # Input normalization
        x_norm = F.normalize(x, p=2, dim=1)

        # --- Hyperbolic pathway ---
        # Project to tangent space
        tangent = self.input_ln(self.input_proj(x_norm))
        # Exponential map to Poincaré ball
        hyp = exp_map(tangent, base=None, c=self.curvature)
        hyp = project_onto_ball(hyp, c=self.curvature)
        # Hyperbolic transform
        hyp = self.hyp_linear(hyp)
        hyp = self.hyp_act(hyp)
        # Back to Euclidean
        hyp_euc = self.hyp_to_euc(hyp)

        # --- Residual pathway ---
        residual = self.residual_proj(x_norm)

        # --- Combine ---
        combined = self.layer_norm(hyp_euc + residual)
        combined = self.act(combined)
        combined = self.dropout(combined)

        # --- Output ---
        z = self.output_proj(combined)

        if return_hidden:
            return z

        return F.normalize(z, p=2, dim=1)

    @property
    def temperature(self):
        return self.log_temperature.exp()


class DeepSafeProjectionHeadManager:
    """Save/load/project utilities for DeepSafe projection head."""

    @staticmethod
    def save(model: DeepSafeProjectionHead, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_dict = {
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "input_dim": model.input_dim,
            "output_dim": model.output_dim,
            "hidden_dim": model.hidden_dim if hasattr(model, 'hidden_dim') else 512,
            "curvature": model.curvature,
            "hyperbolic_dim": model.hyperbolic_dim if hasattr(model, 'hyperbolic_dim') else 128,
        }
        with open(path, "wb") as f:
            pickle.dump(save_dict, f)

    @staticmethod
    def load(path: str, device: str = "cuda") -> DeepSafeProjectionHead:
        with open(path, "rb") as f:
            save_dict = pickle.load(f)

        model = DeepSafeProjectionHead(
            input_dim=save_dict["input_dim"],
            output_dim=save_dict["output_dim"],
            hidden_dim=save_dict.get("hidden_dim", 512),
            curvature=save_dict.get("curvature", 1.0),
            hyperbolic_dim=save_dict.get("hyperbolic_dim", 128),
        )
        model.load_state_dict(save_dict["state_dict"])
        model = model.to(device)
        model.eval()
        return model

    @staticmethod
    def project(
        model: DeepSafeProjectionHead,
        embeddings: np.ndarray,
        batch_size: int = 1024,
        device: str = "cuda",
    ) -> np.ndarray:
        """Project embeddings through the trained projection head."""
        model = model.to(device)
        model.eval()

        projected = []
        with torch.no_grad():
            for i in range(0, len(embeddings), batch_size):
                batch = torch.FloatTensor(embeddings[i: i + batch_size]).to(device)
                z = model(batch)
                projected.append(z.cpu().numpy())

        return np.vstack(projected)
