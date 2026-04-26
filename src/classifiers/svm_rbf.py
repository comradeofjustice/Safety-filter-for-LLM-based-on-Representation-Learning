"""RBF SVM classifier with optional PCA dimensionality reduction."""

import logging
import pickle
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


class RBFSVMClassifier:
    """RBF SVM classifier with optional PCA for high-dimensional data."""

    def __init__(
        self,
        C=1.0,
        gamma="scale",
        class_weight="balanced",
        cache_size=2000,
        probability=True,
        random_state=42,
        pca_components=256,
        use_pca_threshold_dim=2048,
        use_pca_threshold_samples=100000,
        max_samples=50000,
        verbose=True,
    ):
        self.C = C
        self.gamma = gamma
        self.class_weight = class_weight
        self.cache_size = cache_size
        self.probability = probability
        self.random_state = random_state
        self.pca_components = pca_components
        self.use_pca_threshold_dim = use_pca_threshold_dim
        self.use_pca_threshold_samples = use_pca_threshold_samples
        self.max_samples = max_samples
        self.verbose = verbose
        self.model = None
        self.pca = None
        self.use_pca = False
        self.trained = False

    def _should_use_pca(self, X_train: np.ndarray) -> bool:
        """Determine if PCA should be used based on thresholds."""
        n_samples, n_features = X_train.shape
        if n_features >= self.use_pca_threshold_dim:
            logger.info(f"Features ({n_features}) >= threshold ({self.use_pca_threshold_dim}), using PCA")
            return True
        if n_samples >= self.use_pca_threshold_samples:
            logger.info(f"Samples ({n_samples}) >= threshold ({self.use_pca_threshold_samples}), using PCA")
            return True
        return False

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the RBF SVM classifier."""
        if self.max_samples and X_train.shape[0] > self.max_samples:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(X_train.shape[0], size=self.max_samples, replace=False)
            X_train = X_train[idx]
            y_train = y_train[idx]
            logger.info(f"Subsampled to {self.max_samples} samples (stratification not applied)")

        self.use_pca = self._should_use_pca(X_train)

        X_processed = X_train
        if self.use_pca:
            logger.info(f"Applying PCA({self.pca_components})...")
            self.pca = PCA(n_components=self.pca_components, random_state=self.random_state)
            X_processed = self.pca.fit_transform(X_train)
            logger.info(f"PCA reduced from {X_train.shape[1]} to {X_processed.shape[1]} dimensions")

        logger.info(f"Training RBFSVM on {X_processed.shape[0]} samples...")
        self.model = SVC(
            kernel="rbf",
            C=self.C,
            gamma=self.gamma,
            class_weight=self.class_weight,
            cache_size=self.cache_size,
            probability=self.probability,
            random_state=self.random_state,
            verbose=self.verbose,
        )
        self.model.fit(X_processed, y_train)
        self.trained = True
        logger.info("RBFSVM training complete")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels."""
        X_processed = X
        if self.use_pca and self.pca is not None:
            X_processed = self.pca.transform(X)
        return self.model.predict(X_processed)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        X_processed = X
        if self.use_pca and self.pca is not None:
            X_processed = self.pca.transform(X)
        return self.model.predict_proba(X_processed)

    def save(self, path: str):
        """Save model to file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_dict = {
            "model": self.model,
            "pca": self.pca,
            "use_pca": self.use_pca,
        }
        with open(path, "wb") as f:
            pickle.dump(save_dict, f)
        logger.info(f"Saved RBFSVM to {path}")

    @classmethod
    def load(cls, path: str) -> "RBFSVMClassifier":
        """Load model from file."""
        with open(path, "rb") as f:
            save_dict = pickle.load(f)
        clf = cls()
        clf.model = save_dict["model"]
        clf.pca = save_dict["pca"]
        clf.use_pca = save_dict["use_pca"]
        clf.trained = True
        return clf
