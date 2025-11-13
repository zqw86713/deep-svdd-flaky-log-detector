# src/models/deep_svdd.py

import numpy as np
from pyod.models.deep_svdd import DeepSVDD
from typing import Tuple

class DeepSVDDWrapper:
    """Wrapper around PyOD DeepSVDD for log embeddings."""

    def __init__(
        self,
        n_features: int,
        contamination: float = 0.1,
        hidden_neurons: Tuple[int, ...] = (64, 32),
        epochs: int = 50,
        batch_size: int = 64,
    ):
        self.model = DeepSVDD(
            n_features=n_features,
            hidden_neurons=list(hidden_neurons),
            epochs=epochs,
            batch_size=batch_size,
            contamination=contamination,
            verbose=1,
        )

    def fit(self, embeddings: np.ndarray):
        """Fit DeepSVDD on normal log embeddings."""
        self.model.fit(embeddings)

    def decision_scores(self, embeddings: np.ndarray) -> np.ndarray:
        """Return anomaly scores for given embeddings."""
        return self.model.decision_function(embeddings)

    def predict_labels(self, embeddings: np.ndarray) -> np.ndarray:
        """Return binary anomaly labels (1 = anomaly)."""
        return self.model.predict(embeddings)
