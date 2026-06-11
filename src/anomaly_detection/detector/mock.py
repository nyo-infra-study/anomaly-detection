"""Mock detector for development without a trained model."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from anomaly_detection.utils.logging import get_logger

log = get_logger("mock_detector")


class MockDetector:
    """
    Mock detector for testing without a trained model.
    Returns random scores with occasional "anomalies".
    """

    def __init__(self, anomaly_probability: float = 0.1):
        self.anomaly_probability = anomaly_probability

    def load(self) -> None:
        """Simulates model loading."""
        log.info("mock detector ready")

    def predict(self, data: NDArray[Any]) -> NDArray[np.floating[Any]]:
        """
        Generate mock anomaly scores.

        Args:
            data: shape (num_metrics, seq_len)

        Returns:
            scores: shape (num_metrics,) — values in [0, 1]
        """
        num_metrics: int = int(data.shape[0])

        # Generate mostly low scores with occasional high ones
        base_scores = np.random.uniform(0.1, 0.4, size=num_metrics)
        scores: NDArray[np.floating[Any]] = base_scores.astype(np.float32)

        # Randomly make some anomalous
        random_vals = np.random.random(num_metrics)
        anomaly_mask = random_vals < self.anomaly_probability
        num_anomalies = int(np.sum(anomaly_mask))
        if num_anomalies > 0:
            anomaly_values = np.random.uniform(0.8, 0.95, size=num_anomalies)
            scores[anomaly_mask] = anomaly_values.astype(np.float32)

        return scores

    def predict_with_labels(
        self,
        data: NDArray[Any],
        metric_names: list[str],
    ) -> dict[str, float]:
        """Convenience: return dict mapping metric name to score."""
        scores = self.predict(data)
        return dict(zip(metric_names, scores.tolist()))
