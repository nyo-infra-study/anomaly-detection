"""Tests for detector factory."""

from pathlib import Path
from unittest.mock import patch

import pytest

from anomaly_detection.detector import get_detector
from anomaly_detection.detector.hybrid import HybridDetector
from anomaly_detection.detector.mock import MockDetector
from anomaly_detection.detector.statistical import (
    RollingStatisticalDetector,
    StatisticalDetector,
)


class TestGetDetector:
    """Tests for detector factory function."""

    def test_get_detector_mock(self) -> None:
        """Should return MockDetector for 'mock' type."""
        detector = get_detector("mock")
        assert isinstance(detector, MockDetector)

    def test_get_detector_statistical(self) -> None:
        """Should return StatisticalDetector for 'statistical' type."""
        detector = get_detector("statistical")
        assert isinstance(detector, StatisticalDetector)

    def test_get_detector_rolling(self) -> None:
        """Should return RollingStatisticalDetector for 'rolling' type."""
        detector = get_detector("rolling")
        assert isinstance(detector, RollingStatisticalDetector)

    def test_get_detector_hybrid(self) -> None:
        """Should return HybridDetector for 'hybrid' type."""
        detector = get_detector("hybrid")
        assert isinstance(detector, HybridDetector)

    def test_get_detector_timesnet(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should return TimesNetDetector for 'timesnet' type."""
        from anomaly_detection.detector.timesnet import TimesNetDetector

        detector = get_detector("timesnet")
        assert isinstance(detector, TimesNetDetector)

    def test_get_detector_auto_with_model(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should return TimesNet when model file exists."""
        # Create a model file
        model_file = tmp_path / "model.pt"
        model_file.touch()

        monkeypatch.setenv("MODEL_PATH", str(model_file))
        monkeypatch.setenv("DETECTOR_TYPE", "auto")

        from anomaly_detection.config import Settings

        with patch("anomaly_detection.detector.settings", Settings()):
            detector = get_detector("auto")

        from anomaly_detection.detector.timesnet import TimesNetDetector

        assert isinstance(detector, TimesNetDetector)

    def test_get_detector_auto_without_model(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should return Statistical when model file doesn't exist."""
        nonexistent = tmp_path / "nonexistent.pt"
        monkeypatch.setenv("MODEL_PATH", str(nonexistent))
        monkeypatch.setenv("DETECTOR_TYPE", "auto")

        from anomaly_detection.config import Settings

        with patch("anomaly_detection.detector.settings", Settings()):
            detector = get_detector("auto")

        assert isinstance(detector, StatisticalDetector)

    def test_get_detector_uses_settings_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should use settings.detector_type when no argument provided."""
        monkeypatch.setenv("DETECTOR_TYPE", "mock")

        from anomaly_detection.config import Settings

        with patch("anomaly_detection.detector.settings", Settings()):
            detector = get_detector()

        assert isinstance(detector, MockDetector)

    def test_get_detector_statistical_uses_z_threshold(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should pass z_threshold from settings to statistical detector."""
        monkeypatch.setenv("Z_THRESHOLD", "3.5")

        from anomaly_detection.config import Settings

        with patch("anomaly_detection.detector.settings", Settings()):
            detector = get_detector("statistical")

        assert isinstance(detector, StatisticalDetector)
        assert detector.z_threshold == 3.5

    def test_get_detector_rolling_uses_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should pass alpha and z_threshold from settings to rolling detector."""
        monkeypatch.setenv("EMA_ALPHA", "0.2")
        monkeypatch.setenv("Z_THRESHOLD", "3.0")

        from anomaly_detection.config import Settings

        with patch("anomaly_detection.detector.settings", Settings()):
            detector = get_detector("rolling")

        assert isinstance(detector, RollingStatisticalDetector)
        assert detector.alpha == 0.2
        assert detector.z_threshold == 3.0
