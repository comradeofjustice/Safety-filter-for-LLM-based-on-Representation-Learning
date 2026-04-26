"""Linear SVM classifier."""

import logging
import pickle
import os
import numpy as np
from sklearn.svm import LinearSVC

logger = logging.getLogger(__name__)


class LinearSVMClassifier:
    """Linear SVM classifier."""

    def __init__(self, C=1.0, class_weight="balanced", max_iter=5000, random_state=42):
        self.model = LinearSVC(
            C=C,
            class_weight=class_weight,
            max_iter=max_iter,
            random_state=random_state,
        )
        self.trained = False

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier."""
        logger.info(f"Training LinearSVM on {X_train.shape[0]} samples...")
        self.model.fit(X_train, y_train)
        self.trained = True
        logger.info("LinearSVM training complete")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """LinearSVC doesn't support predict_proba directly. Use decision function."""
        # Convert decision function to pseudo-probabilities using sigmoid
        from scipy.special import expit
        decision = self.model.decision_function(X)
        if decision.ndim == 1:
            # Binary case: convert to 2-column probability
            prob_pos = expit(decision)
            return np.vstack([1 - prob_pos, prob_pos]).T
        return decision

    def save(self, path: str):
        """Save model to file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        logger.info(f"Saved LinearSVM to {path}")

    @classmethod
    def load(cls, path: str) -> "LinearSVMClassifier":
        """Load model from file."""
        with open(path, "rb") as f:
            model = pickle.load(f)
        clf = cls()
        clf.model = model
        clf.trained = True
        return clf
