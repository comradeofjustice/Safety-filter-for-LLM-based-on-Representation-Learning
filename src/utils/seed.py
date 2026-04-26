"""Seed utilities for reproducibility."""

import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """Set random seed for all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
