"""Tests for detector modules."""

import numpy as np

from anomaly_detection.detector.mock import MockDetector


class TestMockDetector:
    def test_load(self) -> None:
        detector = MockDetector()
        detector.load()  # Should not raise

    def test_predict_shape(self, sample_data: np.ndarray) -> None:
        detector = MockDetector()
        detector.load()

        scores = detector.predict(sample_data)

        assert scores.shape == (5,)
        assert all(0 <= s <= 1 for s in scores)

    def test_predict_with_labels(
        self, sample_data: np.ndarray, sample_metric_names: list[str]
    ) -> None:
        detector = MockDetector()
        detector.load()

        result = detector.predict_with_labels(sample_data, sample_metric_names)

        assert set(result.keys()) == set(sample_metric_names)
        assert all(isinstance(v, float) for v in result.values())
        assert all(0 <= v <= 1 for v in result.values())

    def test_anomaly_probability(self) -> None:
        """Test that anomaly_probability affects output."""
        # High probability should produce more high scores
        detector_high = MockDetector(anomaly_probability=0.9)
        detector_low = MockDetector(anomaly_probability=0.1)

        np.random.seed(42)
        data = np.random.randn(100, 96).astype(np.float32)

        np.random.seed(42)
        scores_high = detector_high.predict(data)

        np.random.seed(42)
        scores_low = detector_low.predict(data)

        # High probability should have more scores > 0.75
        high_anomalies = (scores_high > 0.75).sum()
        low_anomalies = (scores_low > 0.75).sum()

        assert high_anomalies > low_anomalies
