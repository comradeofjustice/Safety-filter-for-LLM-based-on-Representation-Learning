"""Combined loss functions for DeepSafe: Hierarchical Contrastive + OT + Prototype + Decorrelation.

Loss composition:
  L_total = L_contrastive + lambda_ot * L_OT + lambda_proto * L_prototype + lambda_decorr * L_decorrelation

Where:
  L_contrastive: Hierarchical SupCon loss (binary + intent levels)
  L_OT: Optimal Transport Sinkhorn divergence for class distribution alignment
  L_prototype: Learnable prototype margin loss with intent hierarchy
  L_decorrelation: Barlow Twins spectral decorrelation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al. 2020)."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device
        batch_size = features.shape[0]

        if batch_size < 2:
            return torch.tensor(0.0, device=device)

        labels = labels.contiguous().view(-1, 1)
        mask_pos = torch.eq(labels, labels.T).float().to(device)

        # Remove self-contrast
        mask_self = torch.eye(batch_size, device=device)
        mask_pos = mask_pos - mask_self

        # Compute similarities with temperature
        anchor_dot = torch.matmul(features, features.T) / self.temperature

        # Numerical stability
        logits_max, _ = torch.max(anchor_dot, dim=1, keepdim=True)
        logits = anchor_dot - logits_max.detach()

        # Mask out self
        logits_mask = 1.0 - mask_self

        exp_logits = torch.exp(logits) * logits_mask

        # Log probability
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

        # Mean over positive pairs
        mean_log_prob_pos = (mask_pos * log_prob).sum(1) / (mask_pos.sum(1) + 1e-9)

        loss = -mean_log_prob_pos
        loss = loss[mask_pos.sum(1) > 0].mean()

        return loss if not torch.isnan(loss) else torch.tensor(0.0, device=device)


class PrototypeLoss(nn.Module):
    """Learnable prototype-based margin loss with intent hierarchy.

    Each intent class has a learnable prototype in the projected space.
    The loss:
      - Pulls samples toward their class prototype
      - Pushes samples away from other class prototypes with a margin
      - Applies hierarchical margin: Safe↔Malicious > Safe↔BenignSensitive > BenignSensitive↔Malicious
    """

    def __init__(self, num_classes: int, feature_dim: int, base_margin: float = 0.5):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_classes, feature_dim))
        nn.init.orthogonal_(self.prototypes)
        # Normalize prototypes initially
        self.prototypes.data = F.normalize(self.prototypes.data, dim=1)

        self.base_margin = base_margin
        # Hierarchical margins: safe-vs-malicious > safe-vs-benign > benign-vs-malicious
        self.register_buffer(
            "margin_matrix",
            torch.tensor([
                [0.0, base_margin * 0.5, base_margin],            # safe vs [safe, benign, malicious]
                [base_margin * 0.5, 0.0, base_margin * 0.7],       # benign vs [safe, benign, malicious]
                [base_margin, base_margin * 0.7, 0.0],             # malicious vs [safe, benign, malicious]
            ]),
        )

    def forward(self, features, labels):
        """Compute prototype loss (fully vectorized).

        Args:
            features: (B, D) L2-normalized embeddings
            labels: (B,) integer class labels (0=safe, 1=benign_sensitive, 2=malicious)

        Returns:
            Scalar prototype loss
        """
        device = features.device
        B = features.shape[0]

        if B == 0:
            return torch.tensor(0.0, device=device)

        # Normalize prototypes
        prototypes_norm = F.normalize(self.prototypes, dim=1)

        # Similarity between each sample and each prototype
        sim = torch.matmul(features, prototypes_norm.t())  # (B, num_classes)
        num_classes = sim.shape[1]

        # Attraction: maximize similarity to own prototype
        attraction = -sim[torch.arange(B, device=device), labels].mean()

        # Repulsion: penalize similarity to other prototypes exceeding margin (vectorized)
        # Create margin matrix for each sample in batch
        margins = self.margin_matrix[labels]  # (B, num_classes)
        excess = torch.relu(sim - margins)    # (B, num_classes) - only penalize if above margin

        # Mask out self-class (don't repulse from own prototype)
        mask = torch.ones(B, num_classes, device=device)
        mask[torch.arange(B, device=device), labels] = 0.0

        repulsion = (excess * mask).sum() / (mask.sum() + 1e-9)

        return attraction + 0.3 * repulsion


class HierarchicalContrastiveLoss(nn.Module):
    """Hierarchical contrastive loss with binary + intent levels."""

    def __init__(self, temperature=0.07, alpha=2.0):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.binary_loss = SupConLoss(temperature=temperature)
        self.intent_loss = SupConLoss(temperature=temperature)

    def forward(self, features, binary_labels, intent_labels):
        l_binary = self.binary_loss(features, binary_labels)
        l_intent = self.intent_loss(features, intent_labels)
        return l_binary + self.alpha * l_intent, {"binary": l_binary, "intent": l_intent}


class SinkhornClassLoss(nn.Module):
    """Optimal Transport Sinkhorn divergence for class distribution separation.

    For each class pair (i, j):
      L_OT(i,j) = - SinkhornDivergence(class_i, class_j)  # maximize between-class distance
    Overall:
      L_OT = mean intra-class Sinkhorn - mean inter-class Sinkhorn
    """

    def __init__(self, epsilon=0.05, max_iter=20, reg_intra=0.5):
        super().__init__()
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.reg_intra = reg_intra

    def sinkhorn_div(self, X, Y):
        """Compute Sinkhorn divergence between X and Y efficiently."""
        B_X, B_Y = X.shape[0], Y.shape[0]
        device = X.device

        # Pairwise cost (cosine distance = 1 - cosine_similarity)
        cost = 1.0 - torch.matmul(X, Y.t())

        # Sinkhorn iterations
        K = torch.exp(-cost / self.epsilon)
        u = torch.ones(B_X, device=device) / B_X
        v = torch.ones(B_Y, device=device) / B_Y

        for _ in range(self.max_iter):
            u = 1.0 / (B_X * torch.matmul(K, v) + 1e-12)
            v = 1.0 / (B_Y * torch.matmul(K.t(), u) + 1e-12)

        P = torch.diag(u) @ K @ torch.diag(v)

        return (P * cost).sum()

    def forward(self, features, labels):
        """Compute class-level Sinkhorn regularization loss."""
        device = features.device
        unique_labels = torch.unique(labels)

        if len(unique_labels) < 2:
            return torch.tensor(0.0, device=device)

        # Intra-class compactness
        intra_div = 0.0
        for label in unique_labels:
            mask = labels == label
            if mask.sum() >= 2:
                class_feats = features[mask]
                intra_div += self.sinkhorn_div(class_feats, class_feats)

        intra_div /= max(len(unique_labels), 1)

        # Inter-class separation
        inter_div = 0.0
        label_list = unique_labels.tolist()
        n_pairs = 0
        for i in range(len(label_list)):
            for j in range(i + 1, len(label_list)):
                mask_i = labels == label_list[i]
                mask_j = labels == label_list[j]
                if mask_i.sum() > 0 and mask_j.sum() > 0:
                    inter_div += self.sinkhorn_div(features[mask_i], features[mask_j])
                    n_pairs += 1

        if n_pairs > 0:
            inter_div /= n_pairs

        # We want small intra-class + large inter-class
        return intra_div - self.reg_intra * inter_div


class CounterfactualPairLoss(nn.Module):
    """Repulsion loss for counterfactual pairs.

    Pushes apart paired samples that share keywords but differ in intent.
    Uses a hinge-like loss on cosine similarity.
    """

    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, z_i, z_j):
        """Push apart paired embeddings.

        Args:
            z_i: (N, D) first set
            z_j: (N, D) second set of paired embeddings

        Returns:
            Scalar repulsion loss
        """
        # Cosine similarity should be below margin
        cos_sim = F.cosine_similarity(z_i, z_j, dim=1)
        loss = torch.relu(cos_sim - self.margin).mean()
        return loss


class DeepSafeLoss(nn.Module):
    """Combined DeepSafe loss with all innovations.

    L_total = L_hierarchical + lambda_ot * L_OT + lambda_proto * L_prototype
              + lambda_decorr * L_decorrelation + gamma * L_counterfactual
    """

    def __init__(
        self,
        feature_dim: int = 256,
        temperature: float = 0.07,
        alpha: float = 2.0,
        lambda_ot: float = 0.3,
        lambda_proto: float = 0.2,
        lambda_decorr: float = 0.005,
        gamma: float = 0.5,
        prototype_margin: float = 0.5,
        num_intent_classes: int = 3,
    ):
        super().__init__()
        self.lambda_ot = lambda_ot
        self.lambda_proto = lambda_proto
        self.lambda_decorr = lambda_decorr
        self.gamma = gamma

        self.hierarchical = HierarchicalContrastiveLoss(
            temperature=temperature, alpha=alpha
        )
        self.sinkhorn = SinkhornClassLoss()
        self.prototype = PrototypeLoss(
            num_classes=num_intent_classes,
            feature_dim=feature_dim,
            base_margin=prototype_margin,
        )
        self.counterfactual = CounterfactualPairLoss(margin=0.3)

    def forward(
        self,
        features: torch.Tensor,
        binary_labels: torch.Tensor,
        intent_labels: torch.Tensor,
        cf_pairs: tuple = None,
        return_components: bool = False,
    ):
        """Compute full DeepSafe loss.

        Args:
            features: (B, D) L2-normalized projected embeddings
            binary_labels: (B,) 0=safe, 1=unsafe
            intent_labels: (B,) 0=safe, 1=benign_sensitive, 2=malicious
            cf_pairs: optional (z_i, z_j) tuple for counterfactual pairs
            return_components: if True, return dict of all loss components

        Returns:
            Scalar total loss, or (total_loss, dict) if return_components
        """
        device = features.device

        # 1. Hierarchical contrastive loss (always works)
        l_hier, l_hier_components = self.hierarchical(features, binary_labels, intent_labels)
        l_total = l_hier

        # 2. Sinkhorn OT class regularization (gradient-safe)
        l_ot = torch.tensor(0.0, device=device)
        if self.lambda_ot > 0:
            l_ot = self.sinkhorn(features, intent_labels)
            if not torch.isnan(l_ot):
                l_total = l_total + self.lambda_ot * l_ot

        # 3. Prototype-based margin loss
        l_proto = torch.tensor(0.0, device=device)
        if self.lambda_proto > 0:
            l_proto = self.prototype(features, intent_labels)
            if not torch.isnan(l_proto):
                l_total = l_total + self.lambda_proto * l_proto

        # 4. Spectral decorrelation
        l_decorr = torch.tensor(0.0, device=device)
        if self.lambda_decorr > 0:
            B, D = features.shape
            if B > 1:
                corr = torch.matmul(features.t(), features) / B
                off_diag = corr - torch.diag(torch.diag(corr))
                l_decorr = self.lambda_decorr * (off_diag ** 2).sum() / D
                l_total = l_total + l_decorr

        # 5. Counterfactual pair repulsion
        l_cf = torch.tensor(0.0, device=device)
        if cf_pairs is not None and self.gamma > 0:
            z_i, z_j = cf_pairs
            l_cf = self.gamma * self.counterfactual(z_i, z_j)
            if not torch.isnan(l_cf):
                l_total = l_total + l_cf

        # Clamp to prevent explosion
        if not torch.isnan(l_total):
            l_total = torch.clamp(l_total, max=100.0)
        else:
            l_total = l_hier

        if return_components:
            def safe_val(x):
                if isinstance(x, float):
                    return x if not np.isnan(x) else 0.0
                v = x.item()
                return v if not np.isnan(v) else 0.0

            return l_total, {
                "total": safe_val(l_total),
                "hierarchical": safe_val(l_hier),
                "binary": safe_val(l_hier_components["binary"]),
                "intent": safe_val(l_hier_components["intent"]),
                "sinkhorn_ot": safe_val(l_ot),
                "prototype": safe_val(l_proto),
                "decorrelation": safe_val(l_decorr),
                "counterfactual": safe_val(l_cf),
            }

        return l_total
