"""Metrics utilities for LLM Safety Classifier."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> dict:
    """Compute all classification metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }
    if y_prob is not None:
        y_score = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
    else:
        metrics["roc_auc"] = None
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    return metrics
