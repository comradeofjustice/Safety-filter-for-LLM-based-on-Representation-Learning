"""SafeCL: Intent-Driven Safety Representation Learning via Hierarchical Contrastive Alignment."""

from .projection_head import ProjectionHead, ProjectionHeadManager
from .losses import HierarchicalContrastiveLoss, SupConLoss
from .trainer import SafeCLTrainer
from .intent import IntentAnnotator, IntentTaxonomy
