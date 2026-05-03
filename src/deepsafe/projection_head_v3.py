"""DeepSafe v3 Projection Head: Enhanced hyperbolic + residual + gating + skip connections.

Improvements over v1:
  - Dual hyperbolic layers (v1 had one)
  - Learnable gate α: dynamically balances hyperbolic vs residual pathways
  - Skip connection from input to combined representation
  - GELU activations throughout
  - Larger hidden_dim (768) for 8B encoder support
  - Input projection to 128-dim hyperbolic tangent space (v1 used input_dim directly)
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
    hyperbolic_distance,
    log_map,
)

logger = logging.getLogger(__name__)


class DeepSafeProjectionHeadV3(nn.Module):
    """DeepSafe v3: Enhanced hyperbolic projection head.

    Architecture:
      Input (input_dim) -> L2Norm
        |-- Hyperbolic pathway (2-layer):
        |     Linear(input_dim -> hyp_dim) -> LayerNorm -> exp_map
        |     -> HyperbolicLinear(hyp_dim -> hyp_dim) -> HypAct(GELU)
        |     -> HyperbolicLinear(hyp_dim -> hyp_dim) -> HypAct(GELU)
        |     -> log_map -> HypToEuc(hyp_dim -> hidden_dim, with distance feature)
        |
        |-- Residual pathway:
        |     Linear(input_dim -> hidden_dim)
        |
        |-- Skip connection:
        |     Linear(input_dim -> hidden_dim)(input_norm)
        |
        |-- Gating: α = sigmoid(gate_logit)
        |     combined = α * hyp + (1-α) * residual + skip
        |
        -> LayerNorm -> GELU -> Dropout -> Linear(hidden_dim -> output_dim) -> BN -> L2Norm
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 768,
        output_dim: int = 256,
        hyperbolic_dim: int = 128,
        dropout: float = 0.1,
        curvature: float = 1.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.hyperbolic_dim = hyperbolic_dim
        self.curvature = curvature

        # --- Hyperbolic pathway (deeper: 2 hyp layers) ---
        self.hyp_input_proj = nn.Linear(input_dim, hyperbolic_dim)
        self.hyp_input_ln = nn.LayerNorm(hyperbolic_dim)
        self.hyp_linear1 = HyperbolicLinear(hyperbolic_dim, hyperbolic_dim, c=curvature)
        self.hyp_act1 = HyperbolicActivation(activation=nn.GELU(), c=curvature)
        self.hyp_linear2 = HyperbolicLinear(hyperbolic_dim, hyperbolic_dim, c=curvature)
        self.hyp_act2 = HyperbolicActivation(activation=nn.GELU(), c=curvature)
        self.hyp_to_euc = HyperbolicToEuclidean(hyperbolic_dim, hidden_dim, c=curvature)

        # --- Residual pathway ---
        self.residual_proj = nn.Linear(input_dim, hidden_dim)

        # --- Skip connection from input ---
        self.skip_proj = nn.Linear(input_dim, hidden_dim)

        # --- Learnable gate ---
        self.gate_logit = nn.Parameter(torch.zeros(1))  # starts at α=0.5

        # --- Post-combination processing ---
        self.combine_ln = nn.LayerNorm(hidden_dim)
        self.combine_act = nn.GELU()
        self.combine_dropout = nn.Dropout(dropout)

        # --- Output head ---
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.BatchNorm1d(output_dim),
        )

        # Learnable temperature
        self.log_temperature = nn.Parameter(torch.tensor(0.07).log())

        self._init_weights()

    def _init_weights(self):
        for m in [self.hyp_input_proj, self.residual_proj, self.skip_proj]:
            nn.init.xavier_uniform_(m.weight, gain=0.5)
            nn.init.zeros_(m.bias)

    @property
    def temperature(self):
        return self.log_temperature.exp()

    @property
    def gate_alpha(self):
        return torch.sigmoid(self.gate_logit)

    def forward(self, x: torch.Tensor, return_hidden: bool = False) -> torch.Tensor:
        x_norm = F.normalize(x, p=2, dim=1)

        # --- Hyperbolic pathway ---
        tangent = self.hyp_input_ln(self.hyp_input_proj(x_norm))
        hyp = exp_map(tangent, base=None, c=self.curvature)
        hyp = project_onto_ball(hyp, c=self.curvature)
        hyp = self.hyp_linear1(hyp)
        hyp = self.hyp_act1(hyp)
        hyp = self.hyp_linear2(hyp)
        hyp = self.hyp_act2(hyp)
        hyp_euc = self.hyp_to_euc(hyp)

        # --- Residual pathway ---
        residual = self.residual_proj(x_norm)

        # --- Skip connection ---
        skip = self.skip_proj(x_norm)

        # --- Gated combination ---
        alpha = self.gate_alpha
        combined = alpha * hyp_euc + (1.0 - alpha) * residual + skip

        # --- Post-combination ---
        combined = self.combine_ln(combined)
        combined = self.combine_act(combined)
        combined = self.combine_dropout(combined)

        # --- Output ---
        z = self.output_proj(combined)

        if return_hidden:
            return z

        return F.normalize(z, p=2, dim=1)

    def get_hyp_distance(self, x: torch.Tensor) -> torch.Tensor:
        """Get hyperbolic distance to origin (severity measure)."""
        x_norm = F.normalize(x, p=2, dim=1)
        tangent = self.hyp_input_ln(self.hyp_input_proj(x_norm))
        hyp = exp_map(tangent, base=None, c=self.curvature)
        hyp = project_onto_ball(hyp, c=self.curvature)
        hyp = self.hyp_linear1(hyp)
        hyp = self.hyp_act1(hyp)
        hyp = self.hyp_linear2(hyp)
        hyp = self.hyp_act2(hyp)
        origin = torch.zeros(1, hyp.shape[-1], device=x.device)
        return hyperbolic_distance(hyp, origin.expand(hyp.shape[0], -1), c=self.curvature)


class DeepSafeProjectionHeadV3Manager:
    """Save/load/project utilities for DeepSafe v3 projection head."""

    @staticmethod
    def save(model: DeepSafeProjectionHeadV3, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_dict = {
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "input_dim": model.input_dim,
            "output_dim": model.output_dim,
            "hidden_dim": model.hidden_dim,
            "hyperbolic_dim": model.hyperbolic_dim,
            "curvature": model.curvature,
        }
        with open(path, "wb") as f:
            pickle.dump(save_dict, f)

    @staticmethod
    def load(path: str, device: str = "cuda") -> DeepSafeProjectionHeadV3:
        with open(path, "rb") as f:
            save_dict = pickle.load(f)
        model = DeepSafeProjectionHeadV3(
            input_dim=save_dict["input_dim"],
            output_dim=save_dict["output_dim"],
            hidden_dim=save_dict.get("hidden_dim", 768),
            hyperbolic_dim=save_dict.get("hyperbolic_dim", 128),
            curvature=save_dict.get("curvature", 1.0),
        )
        model.load_state_dict(save_dict["state_dict"])
        model = model.to(device)
        model.eval()
        return model

    @staticmethod
    def project(
        model: DeepSafeProjectionHeadV3,
        embeddings: np.ndarray,
        batch_size: int = 1024,
        device: str = "cuda",
    ) -> np.ndarray:
        model = model.to(device)
        model.eval()
        projected = []
        with torch.no_grad():
            for i in range(0, len(embeddings), batch_size):
                batch = torch.FloatTensor(embeddings[i: i + batch_size]).to(device)
                z = model(batch)
                projected.append(z.cpu().numpy())
        return np.vstack(projected)
