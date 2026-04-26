"""Logistic Regression classifier."""

import logging
import pickle
import os
import numpy as np
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


class LogisticClassifier:
    """Logistic Regression classifier."""

    def __init__(self, max_iter=2000, n_jobs=-1, class_weight="balanced", random_state=42):
        self.model = LogisticRegression(
            max_iter=max_iter,
            n_jobs=n_jobs,
            class_weight=class_weight,
            random_state=random_state,
        )
        self.trained = False

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier."""
        logger.info(f"Training LogisticRegression on {X_train.shape[0]} samples...")
        self.model.fit(X_train, y_train)
        self.trained = True
        logger.info("LogisticRegression training complete")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        return self.model.predict_proba(X)

    def save(self, path: str):
        """Save model to file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        logger.info(f"Saved LogisticRegression to {path}")

    @classmethod
    def load(cls, path: str) -> "LogisticClassifier":
        """Load model from file."""
        with open(path, "rb") as f:
            model = pickle.load(f)
        clf = cls()
        clf.model = model
        clf.trained = True
        return clf
