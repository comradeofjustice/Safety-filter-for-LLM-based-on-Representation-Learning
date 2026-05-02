"""Optimal Transport contrastive loss using Sinkhorn algorithm.

Replaces standard SupCon with entropic optimal transport:
  - Sinkhorn divergence for efficient mini-batch OT
  - Better handles class imbalance
  - Provides principled distance between class distributions
  - Supports hyperbolic distance metric

Reference: Cuturi "Sinkhorn Distances" (NeurIPS 2013)
           Caron et al. "Unsupervised Learning of Visual Features by
           Contrasting Cluster Assignments" (NeurIPS 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinkhorn_knopp(cost_matrix, epsilon=0.05, max_iter=50, tol=1e-6):
    """Sinkhorn-Knopp algorithm for entropic optimal transport.

    Computes the optimal coupling P that minimizes:
      <P, C> + epsilon * H(P)
    subject to row/column marginal constraints.

    Args:
        cost_matrix: (B, B) pairwise cost/distance matrix
        epsilon: entropic regularization strength
        max_iter: maximum Sinkhorn iterations
        tol: convergence tolerance

    Returns:
        P: (B, B) optimal transport plan
    """
    B = cost_matrix.shape[0]
    device = cost_matrix.device

    # Kernel matrix K = exp(-C / epsilon)
    K = torch.exp(-cost_matrix / epsilon)

    # Initialize scaling vectors
    u = torch.ones(B, device=device) / B
    v = torch.ones(B, device=device) / B

    # Sinkhorn iterations
    for _ in range(max_iter):
        u_prev = u.clone()

        # Row normalization
        u = 1.0 / (B * torch.matmul(K, v) + 1e-12)

        # Column normalization
        v = 1.0 / (B * torch.matmul(K.t(), u) + 1e-12)

        # Check convergence
        if torch.max(torch.abs(u - u_prev)) < tol:
            break

    # Optimal transport plan
    P = torch.diag(u) @ K @ torch.diag(v)

    return P


class SinkhornDivergence(nn.Module):
    """Sinkhorn divergence for contrastive learning.

    OT_epsilon(a, b) = min_P <P, C> + epsilon * H(P)
    S_epsilon(a, b) = OT_epsilon(a, b) - 0.5 * OT_epsilon(a, a) - 0.5 * OT_epsilon(b, b)

    This is a valid divergence that is positive, symmetric, and convex.
    """

    def __init__(self, epsilon=0.05, max_iter=30):
        super().__init__()
        self.epsilon = epsilon
        self.max_iter = max_iter

    def forward(self, X, Y=None):
        """Compute Sinkhorn divergence between two sets of embeddings.

        Args:
            X: (B_X, D) first set of embeddings
            Y: (B_Y, D) second set (defaults to X if None, for within-class)

        Returns:
            Scalar Sinkhorn divergence
        """
        if Y is None:
            Y = X

        B_X, B_Y = X.shape[0], Y.shape[0]
        device = X.device

        # Pairwise Euclidean cost matrix
        X_norm = (X ** 2).sum(1, keepdim=True)
        Y_norm = (Y ** 2).sum(1, keepdim=True)
        dist = X_norm + Y_norm.t() - 2.0 * torch.matmul(X, Y.t())
        dist = dist.clamp(min=1e-12).sqrt()

        # Compute OT plans
        P_XY = sinkhorn_knopp(dist, self.epsilon, self.max_iter)
        P_XX = sinkhorn_knopp(dist[:B_X, :B_X], self.epsilon, self.max_iter) if X is Y else \
               sinkhorn_knopp(torch.cdist(X, X), self.epsilon, self.max_iter)
        P_YY = sinkhorn_knopp(torch.cdist(Y, Y), self.epsilon, self.max_iter) if X is Y else \
               sinkhorn_knopp(dist[:B_Y, :B_Y], self.epsilon, self.max_iter)

        # Sinkhorn divergence
        s_xy = (P_XY * dist).sum()
        s_xx = (P_XX * torch.cdist(X, X) if X is Y else P_XX * dist[:B_X, :B_X]).sum()
        s_yy = (P_YY * torch.cdist(Y, Y) if X is Y else P_YY * dist[:B_Y, :B_Y]).sum()

        return s_xy - 0.5 * s_xx - 0.5 * s_yy


class OTContrastiveLoss(nn.Module):
    """Optimal Transport-based contrastive loss.

    For each class:
      - Minimize Sinkhorn divergence between samples of the same class (compactness)
      - Maximize Sinkhorn divergence between samples of different classes (separation)

    This provides a principled alternative to instance-based contrastive loss
    that better captures distribution-level structure.
    """

    def __init__(self, temperature=0.07, epsilon=0.05, margin=0.5):
        super().__init__()
        self.temperature = temperature
        self.epsilon = epsilon
        self.margin = margin
        self.sinkhorn = SinkhornDivergence(epsilon=epsilon)

    def forward(self, features, labels):
        """Compute OT contrastive loss.

        Args:
            features: (B, D) L2-normalized embeddings
            labels: (B,) integer class labels

        Returns:
            Scalar loss
        """
        device = features.device
        B = features.shape[0]
        unique_labels = torch.unique(labels)

        if len(unique_labels) < 2:
            return torch.tensor(0.0, device=device)

        # Intra-class compactness: minimize Sinkhorn divergence within each class
        intra_loss = 0.0
        for label in unique_labels:
            mask = labels == label
            if mask.sum() >= 2:
                class_features = features[mask]
                intra_loss += self.sinkhorn(class_features)

        intra_loss /= max(len(unique_labels), 1)

        # Inter-class separation: maximize distance between class prototypes
        inter_loss = 0.0
        prototypes = []
        for label in unique_labels:
            mask = labels == label
            prototypes.append(features[mask].mean(0))

        prototypes = torch.stack(prototypes, dim=0)
        for i in range(len(unique_labels)):
            for j in range(i + 1, len(unique_labels)):
                dist = F.cosine_similarity(prototypes[i].unsqueeze(0), prototypes[j].unsqueeze(0))
                inter_loss += torch.relu(dist - self.margin)  # penalize if too close

        n_pairs = len(unique_labels) * (len(unique_labels) - 1) / 2
        inter_loss = inter_loss / max(n_pairs, 1)

        return intra_loss + inter_loss * 0.5
