"""Statistical anomaly detector — no ML, just math.

Simple, fast, per-metric anomaly detection using:
- Z-score (how many standard deviations from mean)
- Rolling statistics

Good baseline, works without training.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from anomaly_detection.utils.logging import get_logger

log = get_logger("statistical_detector")


class StatisticalDetector:
    """
    Math-based anomaly detector.

    No training needed — computes anomaly score from statistical properties
    of the input window itself.

    Score formula:
        score = sigmoid(abs(z_score) - threshold_z)

    Where z_score = (last_value - window_mean) / window_std
    """

    def __init__(
        self,
        z_threshold: float = 2.0,
        min_std: float = 1e-6,
    ):
        """
        Args:
            z_threshold: Z-score above this is considered anomalous
            min_std: Minimum std to avoid division by zero
        """
        self.z_threshold = z_threshold
        self.min_std = min_std

    def load(self) -> None:
        """No-op for compatibility with detector interface."""
        log.info("statistical detector ready", z_threshold=self.z_threshold)

    def predict(self, data: NDArray[Any]) -> NDArray[np.floating[Any]]:
        """
        Compute anomaly scores using Z-score.

        Args:
            data: shape (num_metrics, seq_len) — raw or normalized values

        Returns:
            scores: shape (num_metrics,) — values in [0, 1]
        """
        # Get last value of each metric (most recent)
        last_values = data[:, -1]

        # Compute mean and std over the window (excluding last point for fair comparison)
        window = data[:, :-1] if data.shape[1] > 1 else data
        means = window.mean(axis=1)
        stds = window.std(axis=1)
        stds = np.maximum(stds, self.min_std)  # Avoid division by zero

        # Z-score: how many stds away from mean
        z_scores = np.abs((last_values - means) / stds)

        # Convert to [0, 1] score using sigmoid
        # score = sigmoid((z - threshold) * scale)
        # When z = threshold, score = 0.5
        # When z >> threshold, score → 1
        scale = 2.0  # Controls how sharply score rises
        raw_scores = 1 / (1 + np.exp(-(z_scores - self.z_threshold) * scale))

        return raw_scores.astype(np.float32)

    def predict_with_labels(
        self,
        data: NDArray[Any],
        metric_names: list[str],
    ) -> dict[str, float]:
        """Convenience: return dict mapping metric name to score."""
        scores = self.predict(data)
        return dict(zip(metric_names, scores.tolist()))


class RollingStatisticalDetector:
    """
    Rolling window statistical detector.

    Maintains history across inference cycles for more stable baselines.
    Uses exponential moving average (EMA) for mean and std.
    """

    def __init__(
        self,
        alpha: float = 0.1,  # EMA smoothing factor
        z_threshold: float = 2.5,
        min_std: float = 1e-6,
    ):
        self.alpha = alpha
        self.z_threshold = z_threshold
        self.min_std = min_std

        # State: EMA of mean and variance per metric
        self.ema_mean: dict[str, float] = {}
        self.ema_var: dict[str, float] = {}

    def load(self) -> None:
        """No-op for compatibility."""
        log.info(
            "rolling statistical detector ready",
            alpha=self.alpha,
            z_threshold=self.z_threshold,
        )

    def predict_with_labels(
        self,
        data: NDArray[Any],
        metric_names: list[str],
    ) -> dict[str, float]:
        """
        Compute scores and update rolling statistics.

        Args:
            data: shape (num_metrics, seq_len)
            metric_names: list of metric identifiers
        """
        scores: dict[str, float] = {}

        for i, name in enumerate(metric_names):
            series = data[i]
            current_mean = float(series.mean())
            current_var = float(series.var())
            last_value = float(series[-1])

            # Initialize or update EMA
            if name not in self.ema_mean:
                self.ema_mean[name] = current_mean
                self.ema_var[name] = current_var
            else:
                self.ema_mean[name] = (
                    self.alpha * current_mean + (1 - self.alpha) * self.ema_mean[name]
                )
                self.ema_var[name] = (
                    self.alpha * current_var + (1 - self.alpha) * self.ema_var[name]
                )

            # Compute z-score against rolling baseline
            ema_std = max(np.sqrt(self.ema_var[name]), self.min_std)
            z_score = abs(last_value - self.ema_mean[name]) / ema_std

            # Convert to [0, 1]
            score = 1 / (1 + np.exp(-(z_score - self.z_threshold) * 2.0))
            scores[name] = float(score)

        return scores

    def predict(self, data: NDArray[Any]) -> NDArray[np.floating[Any]]:
        """For interface compatibility — generates temporary names."""
        names = [f"metric_{i}" for i in range(data.shape[0])]
        scores_dict = self.predict_with_labels(data, names)
        return np.array(list(scores_dict.values()), dtype=np.float32)
