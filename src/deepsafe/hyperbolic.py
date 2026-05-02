"""Hyperbolic geometry operations in the Poincaré ball model.

Key operations for neural networks in hyperbolic space:
  - Exponential map (tangent -> manifold)
  - Logarithmic map (manifold -> tangent)
  - Möbius addition (gyrovector addition)
  - Möbius matrix-vector multiplication
  - Hyperbolic distance

Reference: Ganea et al. "Hyperbolic Neural Networks" (NeurIPS 2018)
           Nickel & Kiela "Poincaré Embeddings" (NeurIPS 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


EPSILON = 1e-5  # Must be > 1.2e-7 for float32 safety (1.0 - 1e-8 rounds to 1.0 in fp32)


def poincare_ball_grad(c=1.0):
    """Conformal factor for Poincaré ball with curvature -c.

    lambda_x = 2 / (1 - c * ||x||^2)
    """
    def lambda_x(x):
        return 2.0 / (1.0 - c * torch.sum(x ** 2, dim=-1, keepdim=True)).clamp(min=EPSILON)
    return lambda_x


def exp_map(x, base=None, c=1.0):
    """Exponential map from tangent space at `base` to Poincaré ball.

    Maps a vector v in the tangent space T_base M to the manifold M.

    Args:
        x: (..., dim) vectors in tangent space
        base: (..., dim) base point on manifold (default: origin)
        c: curvature parameter (default: 1.0 for unit ball)

    Returns:
        (..., dim) points on Poincaré ball
    """
    if base is None:
        # At origin: exp_0(v) = tanh(sqrt(c) * ||v||) * v / (sqrt(c) * ||v||)
        v_norm = torch.norm(x, dim=-1, keepdim=True).clamp(min=EPSILON)
        sqrt_c = c ** 0.5
        scale = torch.tanh(sqrt_c * v_norm) / (sqrt_c * v_norm)
        return scale * x

    # General case with Möbius addition
    lam = poincare_ball_grad(c)
    lambda_base = lam(base)
    v_norm = torch.norm(x, dim=-1, keepdim=True).clamp(min=EPSILON)
    sqrt_c = c ** 0.5
    factor = torch.tanh(sqrt_c * lambda_base * v_norm / 2.0) / (sqrt_c * v_norm)
    return mobius_add(base, factor * x, c=c)


def log_map(x, base=None, c=1.0):
    """Logarithmic map from Poincaré ball to tangent space.

    Args:
        x: (..., dim) points on manifold
        base: (..., dim) base point (default: origin)
        c: curvature parameter

    Returns:
        (..., dim) vectors in tangent space
    """
    if base is None:
        v_norm = torch.norm(x, dim=-1, keepdim=True).clamp(min=EPSILON)
        sqrt_c = c ** 0.5
        scale = torch.atanh(sqrt_c * v_norm.clamp(max=1.0/sqrt_c - EPSILON)) / (sqrt_c * v_norm)
        return scale * x

    lam = poincare_ball_grad(c)
    lambda_base = lam(base)
    diff = mobius_add(-base, x, c=c)
    diff_norm = torch.norm(diff, dim=-1, keepdim=True).clamp(min=EPSILON)
    sqrt_c = c ** 0.5
    scale = 2.0 * torch.atanh(sqrt_c * diff_norm.clamp(max=1.0/sqrt_c - EPSILON)) / (sqrt_c * lambda_base * diff_norm)
    return scale * diff


def mobius_add(x, y, c=1.0):
    """Möbius addition in the Poincaré ball.

    x ⊕_c y = ( (1 + 2c<x,y> + c||y||^2)x + (1 - c||x||^2)y ) / (1 + 2c<x,y> + c^2||x||^2||y||^2)

    Args:
        x: (..., dim) first operand
        y: (..., dim) second operand
        c: curvature parameter

    Returns:
        (..., dim) result of Möbius addition
    """
    x_norm_sq = torch.sum(x ** 2, dim=-1, keepdim=True)
    y_norm_sq = torch.sum(y ** 2, dim=-1, keepdim=True)
    xy_inner = torch.sum(x * y, dim=-1, keepdim=True)

    numerator = (1.0 + 2.0 * c * xy_inner + c * y_norm_sq) * x + (1.0 - c * x_norm_sq) * y
    denominator = 1.0 + 2.0 * c * xy_inner + c ** 2 * x_norm_sq * y_norm_sq

    result = numerator / denominator.clamp(min=EPSILON)
    return project_onto_ball(result, c=c)


def mobius_matvec(m, x, c=1.0):
    """Möbius matrix-vector multiplication.

    m ⊗_c x = exp_0(m · log_0(x))

    Args:
        m: (..., out_dim, in_dim) linear transformation matrix
        x: (..., in_dim) point on Poincaré ball
        c: curvature parameter

    Returns:
        (..., out_dim) transformed point on Poincaré ball
    """
    # Log map to tangent space at origin
    x_tangent = log_map(x, base=None, c=c)
    # Linear transform in tangent space
    if m.dim() == 2 and x_tangent.dim() == 2:
        mx = F.linear(x_tangent, m)
    elif m.dim() == 1:
        mx = m * x_tangent
    else:
        mx = torch.matmul(m, x_tangent.unsqueeze(-1)).squeeze(-1)
    # Exponential map back to manifold
    return exp_map(mx, base=None, c=c)


def hyperbolic_distance(x, y, c=1.0):
    """Distance between two points in the Poincaré ball.

    d_c(x, y) = (2 / sqrt(c)) * arctanh(sqrt(c) * ||-x ⊕_c y||)

    Args:
        x: (..., dim) first point
        y: (..., dim) second point (or same shape as x)
        c: curvature parameter

    Returns:
        (...,) distances
    """
    diff = mobius_add(-x, y, c=c)
    diff_norm = torch.norm(diff, dim=-1).clamp(max=1.0 / (c ** 0.5) - EPSILON)
    return (2.0 / (c ** 0.5)) * torch.atanh((c ** 0.5) * diff_norm)


def project_onto_ball(x, c=1.0, eps=EPSILON):
    """Project points back onto the Poincaré ball if they escape."""
    norm = torch.norm(x, dim=-1, keepdim=True)
    max_norm = (1.0 / (c ** 0.5)) - eps
    scale = torch.where(norm > max_norm, max_norm / norm, torch.ones_like(norm))
    return x * scale


class HyperbolicLinear(nn.Module):
    """Linear layer in the Poincaré ball using Möbius matrix-vector multiplication.

    h ⊗_c x ⊕_c b

    where h is the linear transform, x is the input point, b is the bias.
    """

    def __init__(self, in_features, out_features, c=1.0, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.c = c

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight, gain=0.5)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        # Möbius matrix-vector multiplication
        out = mobius_matvec(self.weight, x, c=self.c)
        # Möbius add bias
        if self.bias is not None:
            out = mobius_add(out, self.bias.unsqueeze(0).expand(out.shape[0], -1), c=self.c)
        return out


class HyperbolicActivation(nn.Module):
    """Apply activation in the tangent space."""

    def __init__(self, activation=nn.ReLU(), c=1.0):
        super().__init__()
        self.activation = activation
        self.c = c

    def forward(self, x):
        # Map to tangent space at origin
        tangent = log_map(x, base=None, c=self.c)
        # Apply activation
        tangent = self.activation(tangent)
        # Map back to manifold
        return exp_map(tangent, base=None, c=self.c)


class HyperbolicToEuclidean(nn.Module):
    """Bridge from hyperbolic space back to Euclidean for downstream classifier.

    Concatenates [log_0(x), hyperbolic_distance_to_origin(x)] to capture
    both the direction and the magnitude (hierarchy level) in hyperbolic space.
    """

    def __init__(self, input_dim, output_dim, c=1.0):
        super().__init__()
        self.c = c
        self.linear = nn.Linear(input_dim + 1, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        # Log map gives us Euclidean representation
        euclidean = log_map(x, base=None, c=self.c)
        # Distance to origin captures hierarchical level
        origin = torch.zeros(1, x.shape[-1], device=x.device)
        dist = hyperbolic_distance(x, origin.expand(x.shape[0], -1), c=self.c)
        # Concatenate and transform
        combined = torch.cat([euclidean, dist.unsqueeze(-1)], dim=-1)
        return self.layer_norm(self.linear(combined))
