#!/usr/bin/env python
"""scripts/05_evaluate.py - Evaluate classifiers on test data."""

import argparse
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.classifiers.linear import LogisticClassifier
from src.classifiers.mlp import MLPClassifier
from src.classifiers.svm_linear import LinearSVMClassifier
from src.classifiers.svm_rbf import RBFSVMClassifier
from src.utils.metrics import compute_metrics
from src.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CLASSIFIERS = {
    "logistic": LogisticClassifier,
    "mlp": MLPClassifier,
    "linear_svm": LinearSVMClassifier,
    "rbf_svm": RBFSVMClassifier,
}


def plot_confusion_matrix(cm, save_path: str, title: str = "Confusion Matrix"):
    """Plot and save confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)

    classes = ["Safe (0)", "Unsafe (1)"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved confusion matrix to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate classifiers")
    parser.add_argument(
        "--embed-model",
        type=str,
        required=True,
        help="Embedding model name",
    )
    parser.add_argument(
        "--classifiers",
        type=str,
        nargs="+",
        required=True,
        choices=list(CLASSIFIERS.keys()),
        help="Classifier types to evaluate",
    )
    args = parser.parse_args()

    set_seed(42)

    embed_dir = f"embeddings/{args.embed_model}"
    model_dir = f"models/{args.embed_model}"
    figures_dir = "reports/figures"
    os.makedirs(figures_dir, exist_ok=True)

    # Load test embeddings
    X_test = np.load(os.path.join(embed_dir, "test.npy"))
    y_test = np.load(os.path.join(embed_dir, "test_labels.npy"))
    logger.info(f"Loaded test embeddings: {X_test.shape}")

    # Evaluate each classifier
    metrics_records = []

    for clf_name in args.classifiers:
        model_path = os.path.join(model_dir, f"{clf_name}.pkl")
        if not os.path.exists(model_path):
            logger.warning(f"Model not found: {model_path}, skipping")
            continue

        logger.info(f"Evaluating {clf_name}...")

        # Load model
        clf_class = CLASSIFIERS[clf_name]
        clf = clf_class.load(model_path)

        # Predict
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)

        # Compute metrics
        metrics = compute_metrics(y_test, y_pred, y_prob)

        # Record metrics
        for metric_name, value in metrics.items():
            if metric_name == "confusion_matrix":
                continue
            metrics_records.append(
                {
                    "embed_model": args.embed_model,
                    "classifier": clf_name,
                    "metric": metric_name,
                    "value": value,
                }
            )

        # Plot confusion matrix
        cm_path = os.path.join(figures_dir, f"{args.embed_model}_{clf_name}_cm.png")
        cm = np.array(metrics["confusion_matrix"])
        plot_confusion_matrix(cm, cm_path, title=f"{clf_name} - {args.embed_model}")

        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  F1 (macro): {metrics['f1_macro']:.4f}")
        logger.info(f"  ROC AUC: {metrics['roc_auc']:.4f}")

    # Save metrics to CSV
    os.makedirs("reports", exist_ok=True)
    metrics_df = pd.DataFrame(metrics_records)
    metrics_df.to_csv("reports/metrics.csv", index=False)
    logger.info(f"Saved metrics to reports/metrics.csv")
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
