# src/models/deep_svdd.py

import numpy as np
from typing import Optional


class SimpleSVDD:
    """
    A simple Deep SVDD-style anomaly detector on top of embeddings.

    - We assume input X are latent vectors from a neural encoder.
    - We fit a center c using normal samples.
    - Anomaly score = squared distance to center ||x - c||^2.
    - Optionally we set a radius based on a quantile (nu) to decide inlier/outlier.
    """

    def __init__(self, nu: float = 0.1):
        """
        Args:
            nu: expected fraction of anomalies (0 < nu < 1).
                 e.g., nu=0.1 means we expect ~10% anomalies.
        """
        self.nu = nu
        self.center_: Optional[np.ndarray] = None
        self.radius2_: Optional[float] = None

    def fit(self, X_normal: np.ndarray):
        """
        Fit the center and radius on normal samples.
        X_normal: shape (N, D)
        """
        # Center = mean of normal embeddings
        self.center_ = X_normal.mean(axis=0)

        # Squared distances to center
        d2 = np.sum((X_normal - self.center_) ** 2, axis=1)

        # Radius^2 = (1 - nu) quantile of distances
        q = 1.0 - self.nu
        self.radius2_ = np.quantile(d2, q)

    def decision_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores for X.
        Score = squared distance to center (larger = more anomalous).
        """
        if self.center_ is None:
            raise RuntimeError("Model is not fitted yet.")
        d2 = np.sum((X - self.center_) ** 2, axis=1)
        return d2

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary anomaly label: 1 = anomaly, 0 = normal.
        Uses radius^2 as threshold.
        """
        scores = self.decision_scores(X)
        if self.radius2_ is None:
            raise RuntimeError("Model radius is not set. Call fit() first.")
        return (scores > self.radius2_).astype(int)
