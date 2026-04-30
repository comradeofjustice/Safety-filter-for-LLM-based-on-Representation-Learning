"""Hierarchical contrastive loss for intent-aware safety representation learning.

Design:
  L_total = L_binary + alpha * L_intent + beta * L_finegrained

Where:
  L_binary: SupCon loss with safe(0) vs unsafe(1) labels
  L_intent: SupCon loss with safe vs benign_sensitive vs malicious (3-class)
  L_finegrained: SupCon loss preserving fine-grained attack-type structure within malicious

Reference: Khosla et al. "Supervised Contrastive Learning" (NeurIPS 2020)
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al. 2020).

    For each anchor, pulls together embeddings with the same label
    and pushes apart embeddings with different labels.

    Args:
        temperature: Temperature parameter (lower = harder contrast)
        base_temperature: Base temperature for scaling
    """

    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(
        self, features: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            features: (batch_size, dim) L2-normalized embeddings
            labels: (batch_size,) integer class labels
            mask: Optional (batch_size, batch_size) boolean mask for valid pairs

        Returns:
            Scalar loss
        """
        device = features.device
        batch_size = features.shape[0]

        if batch_size < 2:
            return torch.tensor(0.0, device=device)

        # Build label mask: positive pairs share the same label
        labels = labels.contiguous().view(-1, 1)
        mask_positive = torch.eq(labels, labels.T).float().to(device)

        # Remove self-contrast (diagonal)
        logits_mask = torch.ones((batch_size, batch_size), device=device)
        logits_mask = logits_mask - torch.eye(batch_size, device=device)

        # Apply external mask if provided
        if mask is not None:
            mask_positive = mask_positive * mask.float()
            logits_mask = logits_mask * mask.float()

        # Compute similarity (dot product since features are L2-normalized)
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T), self.temperature
        )

        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Mask out self and invalid pairs
        logits_mask_neg = 1 - torch.eye(batch_size, device=device)
        if mask is not None:
            logits_mask_neg = logits_mask_neg * mask.float()

        exp_logits = torch.exp(logits) * logits_mask_neg

        # Log probability
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

        # Mean over positive pairs
        mean_log_prob_pos = (mask_positive * log_prob).sum(1) / (
            mask_positive.sum(1) + 1e-9
        )

        # Loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss[mask_positive.sum(1) > 0].mean()

        return loss if not torch.isnan(loss) else torch.tensor(0.0, device=device)


class HierarchicalContrastiveLoss(nn.Module):
    """Hierarchical contrastive loss for intent-driven safety representation.

    Three levels:
      L1: Binary safe/unsafe separation
      L2: Intent-level separation (safe / benign_sensitive / malicious)
      L3: Fine-grained structure preservation within malicious class

    L_total = w1 * L_binary + w2 * L_intent + w3 * L_finegrained
    """

    def __init__(
        self,
        temperature: float = 0.07,
        alpha: float = 0.5,  # weight for L_intent
        beta: float = 0.3,   # weight for L_finegrained
        use_finegrained: bool = True,
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.use_finegrained = use_finegrained

        self.binary_loss = SupConLoss(temperature=temperature)
        self.intent_loss = SupConLoss(temperature=temperature)
        if use_finegrained:
            self.finegrained_loss = SupConLoss(temperature=temperature * 0.5)

    def forward(
        self,
        features: torch.Tensor,
        binary_labels: torch.Tensor,
        intent_labels: torch.Tensor,
        finegrained_labels: torch.Tensor = None,
    ) -> dict:
        """
        Args:
            features: (batch_size, dim) L2-normalized embeddings
            binary_labels: (batch_size,) 0=safe, 1=unsafe
            intent_labels: (batch_size,) 0=safe, 1=benign_sensitive, 2=malicious
            finegrained_labels: (batch_size,) optional fine-grained attack type labels

        Returns:
            dict with keys: 'total', 'binary', 'intent', 'finegrained'
        """
        device = features.device

        # L1: Binary contrastive (safe vs unsafe)
        l_binary = self.binary_loss(features, binary_labels)

        # L2: Intent-level contrastive (3-class)
        l_intent = self.intent_loss(features, intent_labels)

        # L3: Fine-grained structure within malicious (optional)
        l_finegrained = torch.tensor(0.0, device=device)
        if self.use_finegrained and finegrained_labels is not None:
            # Only apply fine-grained loss to malicious samples
            malicious_mask = (intent_labels == 2).nonzero(as_tuple=True)[0]
            if len(malicious_mask) >= 2:
                mal_features = features[malicious_mask]
                mal_fine_labels = finegrained_labels[malicious_mask]
                l_finegrained = self.finegrained_loss(mal_features, mal_fine_labels)

        # Combine losses
        l_total = l_binary + self.alpha * l_intent
        if self.use_finegrained:
            l_total = l_total + self.beta * l_finegrained

        return {
            "total": l_total,
            "binary": l_binary,
            "intent": l_intent,
            "finegrained": l_finegrained,
        }


class IntentAwareContrastiveLoss(nn.Module):
    """Alternative formulation: intent-weighted contrastive loss.

    Assigns higher weight to confusing the model about intent pairs
    (benign_sensitive vs malicious) than safe vs malicious pairs.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        hard_negative_weight: float = 2.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.hard_negative_weight = hard_negative_weight

    def forward(
        self,
        features: torch.Tensor,
        intent_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Weighted contrastive loss that emphasizes hard negative pairs.

        Benign-sensitive samples that are close to malicious samples in the
        original embedding space get higher weight.
        """
        device = features.device
        batch_size = features.shape[0]

        if batch_size < 2:
            return torch.tensor(0.0, device=device)

        # Compute pairwise similarities
        sim = torch.matmul(features, features.T) / self.temperature

        # Positive mask (same intent class)
        labels = intent_labels.contiguous().view(-1, 1)
        pos_mask = torch.eq(labels, labels.T).float().to(device)
        pos_mask = pos_mask - torch.eye(batch_size, device=device)  # remove self

        # Hard negative mask: benign_sensitive (1) vs malicious (2)
        hard_neg_mask = torch.zeros((batch_size, batch_size), device=device)
        for i in range(batch_size):
            for j in range(batch_size):
                if i != j:
                    if (intent_labels[i] == 1 and intent_labels[j] == 2) or \
                       (intent_labels[i] == 2 and intent_labels[j] == 1):
                        hard_neg_mask[i, j] = self.hard_negative_weight
                    elif intent_labels[i] != intent_labels[j]:
                        hard_neg_mask[i, j] = 1.0

        # Apply weighting
        neg_mask = (1.0 - pos_mask) - torch.eye(batch_size, device=device)
        neg_mask = neg_mask.clamp(min=0)  # ensure non-negative

        # Numerical stability
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # Compute loss
        exp_sim = torch.exp(sim)
        pos_sum = (exp_sim * pos_mask).sum(1)
        neg_sum = (exp_sim * neg_mask * hard_neg_mask).sum(1)

        loss = -torch.log((pos_sum + 1e-9) / (pos_sum + neg_sum + 1e-9))
        loss = loss[pos_mask.sum(1) > 0].mean()

        return loss if not torch.isnan(loss) else torch.tensor(0.0, device=device)
