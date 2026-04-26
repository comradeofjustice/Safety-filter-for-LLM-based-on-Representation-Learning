#!/usr/bin/env python
"""scripts/04_train.py - Train classifiers on embedded data."""

import argparse
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.classifiers.linear import LogisticClassifier
from src.classifiers.mlp import MLPClassifier
from src.classifiers.svm_linear import LinearSVMClassifier
from src.classifiers.svm_rbf import RBFSVMClassifier
from src.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CLASSIFIERS = {
    "logistic": LogisticClassifier,
    "mlp": MLPClassifier,
    "linear_svm": LinearSVMClassifier,
    "rbf_svm": RBFSVMClassifier,
}


def main():
    parser = argparse.ArgumentParser(description="Train classifiers on embedded data")
    parser.add_argument(
        "--embed-model",
        type=str,
        required=True,
        help="Embedding model name (e.g., qwen3-embedding-0.6B)",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        required=True,
        choices=list(CLASSIFIERS.keys()),
        help="Classifier type",
    )
    args = parser.parse_args()

    set_seed(42)

    embed_dir = f"embeddings/{args.embed_model}"
    model_dir = f"models/{args.embed_model}"
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, f"{args.classifier}.pkl")

    # Check if model already exists
    if os.path.exists(model_path):
        logger.info(f"Model already exists at {model_path}, skipping. Delete to retrain.")
        return

    # Load embeddings
    X_train = np.load(os.path.join(embed_dir, "train.npy"))
    y_train = np.load(os.path.join(embed_dir, "train_labels.npy"))
    logger.info(f"Loaded train embeddings: {X_train.shape}")

    # Initialize classifier
    clf_class = CLASSIFIERS[args.classifier]
    clf = clf_class()

    # Train
    clf.train(X_train, y_train)

    # Save
    clf.save(model_path)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
