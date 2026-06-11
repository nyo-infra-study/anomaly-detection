"""Tests for statistical detector."""

import numpy as np

from anomaly_detection.detector.statistical import (
    RollingStatisticalDetector,
    StatisticalDetector,
)


class TestStatisticalDetector:
    def test_load(self) -> None:
        detector = StatisticalDetector()
        detector.load()  # Should not raise

    def test_predict_normal_data(self) -> None:
        """Normal data should have low scores."""
        detector = StatisticalDetector(z_threshold=2.0)
        detector.load()

        # Create normal data: mean=100, std=10, last value within 1 std
        np.random.seed(42)
        data = np.random.normal(100, 10, size=(3, 96)).astype(np.float32)
        # Set last values to be close to mean
        data[:, -1] = [100, 102, 98]

        scores = detector.predict(data)

        assert scores.shape == (3,)
        # All scores should be low (< 0.5) for normal data
        assert all(s < 0.5 for s in scores)

    def test_predict_anomalous_data(self) -> None:
        """Anomalous spike should have high score."""
        detector = StatisticalDetector(z_threshold=2.0)
        detector.load()

        # Create data with one anomalous metric
        np.random.seed(42)
        data = np.random.normal(100, 10, size=(3, 96)).astype(np.float32)
        # Set last value of first metric to be 5 std away
        data[0, -1] = 150  # 5 std away from mean

        scores = detector.predict(data)

        # First metric should have high score
        assert scores[0] > 0.8
        # Others should be normal
        assert scores[1] < 0.5
        assert scores[2] < 0.5

    def test_predict_with_labels(self) -> None:
        detector = StatisticalDetector()
        detector.load()

        data = np.random.randn(3, 96).astype(np.float32)
        names = ["cpu", "memory", "disk"]

        result = detector.predict_with_labels(data, names)

        assert set(result.keys()) == {"cpu", "memory", "disk"}
        assert all(0 <= v <= 1 for v in result.values())

    def test_constant_series(self) -> None:
        """Constant series should not crash (edge case)."""
        detector = StatisticalDetector()
        detector.load()

        # All same values
        data = np.ones((2, 96), dtype=np.float32) * 100

        scores = detector.predict(data)

        # Should not be NaN
        assert not np.isnan(scores).any()


class TestRollingStatisticalDetector:
    def test_load(self) -> None:
        detector = RollingStatisticalDetector()
        detector.load()

    def test_maintains_state(self) -> None:
        """Rolling detector should maintain EMA state."""
        detector = RollingStatisticalDetector(alpha=0.5)
        detector.load()

        # First call
        data1 = np.ones((1, 96), dtype=np.float32) * 100
        names = ["metric_a"]
        detector.predict_with_labels(data1, names)

        assert "metric_a" in detector.ema_mean
        assert detector.ema_mean["metric_a"] == 100.0

        # Second call with different mean
        data2 = np.ones((1, 96), dtype=np.float32) * 200
        detector.predict_with_labels(data2, names)

        # EMA should be updated: 0.5 * 200 + 0.5 * 100 = 150
        assert detector.ema_mean["metric_a"] == 150.0

    def test_spike_detection(self) -> None:
        """Rolling detector should detect sudden spikes."""
        detector = RollingStatisticalDetector(alpha=0.1, z_threshold=2.0)
        detector.load()

        names = ["metric_a"]

        # Warm up with normal data
        for _ in range(10):
            normal_data = np.random.normal(100, 10, size=(1, 96)).astype(np.float32)
            detector.predict_with_labels(normal_data, names)

        # Now send a spike
        spike_data = np.random.normal(100, 10, size=(1, 96)).astype(np.float32)
        spike_data[0, -1] = 200  # Big spike at the end

        scores = detector.predict_with_labels(spike_data, names)

        # Should detect the spike
        assert scores["metric_a"] > 0.7
