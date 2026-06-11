"""Tests for hybrid detector."""

import numpy as np
import pytest

from anomaly_detection.detector.hybrid import HybridDetector


class TestHybridDetector:
    """Tests for HybridDetector routing logic."""

    def test_init_with_critical_services_list(self) -> None:
        """Should accept critical services as a list."""
        detector = HybridDetector(critical_services=["checkout", "payments"])
        assert detector.critical_services == {"checkout", "payments"}

    def test_init_with_empty_critical_services(self) -> None:
        """Should handle empty critical services."""
        detector = HybridDetector(critical_services=[])
        assert detector.critical_services == set()

    def test_extract_service_name_with_service_label(self) -> None:
        """Should extract service name from service= label."""
        detector = HybridDetector(critical_services=[])
        metric = 'http_requests_total{service="checkout",pod="checkout-abc123"}'
        assert detector._extract_service_name(metric) == "checkout"

    def test_extract_service_name_with_job_label(self) -> None:
        """Should fall back to job= label if service= not found."""
        detector = HybridDetector(critical_services=[])
        metric = 'http_requests_total{job="payments",instance="10.0.0.1:8080"}'
        assert detector._extract_service_name(metric) == "payments"

    def test_extract_service_name_without_labels(self) -> None:
        """Should return full metric name if no service/job label."""
        detector = HybridDetector(critical_services=[])
        metric = "node_cpu_seconds_total"
        assert detector._extract_service_name(metric) == "node_cpu_seconds_total"

    def test_is_critical_returns_true_for_critical_service(self) -> None:
        """Should identify critical service metrics."""
        detector = HybridDetector(critical_services=["checkout", "payments"])
        metric = 'http_latency{service="checkout"}'
        assert detector._is_critical(metric) is True

    def test_is_critical_returns_false_for_non_critical_service(self) -> None:
        """Should return false for non-critical services."""
        detector = HybridDetector(critical_services=["checkout", "payments"])
        metric = 'http_latency{service="newsletter"}'
        assert detector._is_critical(metric) is False

    def test_load_skips_critical_detector_when_no_critical_services(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should only load default detector when no critical services."""
        detector = HybridDetector(
            critical_services=[],
            default_detector_type="mock",
        )
        detector.load()

        assert detector.critical_detector is None
        assert detector.default_detector is not None

    def test_load_creates_both_detectors_when_critical_services_exist(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should load both detectors when critical services are configured."""
        detector = HybridDetector(
            critical_services=["checkout"],
            critical_detector_type="mock",  # Use mock to avoid TimesNet dependency
            default_detector_type="mock",
        )
        detector.load()

        assert detector.critical_detector is not None
        assert detector.default_detector is not None

    def test_predict_with_labels_routes_to_correct_detector(self) -> None:
        """Should route metrics to appropriate detectors."""
        detector = HybridDetector(
            critical_services=["checkout"],
            critical_detector_type="mock",
            default_detector_type="mock",
        )
        detector.load()

        # Create test data: 3 metrics, 10 time steps
        data = np.random.randn(3, 10).astype(np.float32)
        metric_names = [
            'latency{service="checkout"}',  # critical
            'latency{service="newsletter"}',  # default
            'cpu{job="worker"}',  # default
        ]

        results = detector.predict_with_labels(data, metric_names)

        # Check all metrics are in results
        assert len(results) == 3

        # Check routing
        assert results['latency{service="checkout"}']["detector"] == "mock"
        assert results['latency{service="checkout"}']["service"] == "checkout"

        assert results['latency{service="newsletter"}']["detector"] == "mock"
        assert results['latency{service="newsletter"}']["service"] == "newsletter"

        assert results['cpu{job="worker"}']["detector"] == "mock"
        assert results['cpu{job="worker"}']["service"] == "worker"

    def test_predict_with_labels_returns_scores(self) -> None:
        """Should return valid scores for all metrics."""
        detector = HybridDetector(
            critical_services=["checkout"],
            critical_detector_type="mock",
            default_detector_type="mock",
        )
        detector.load()

        data = np.random.randn(2, 10).astype(np.float32)
        metric_names = [
            'latency{service="checkout"}',
            'latency{service="other"}',
        ]

        results = detector.predict_with_labels(data, metric_names)

        for name in metric_names:
            assert "score" in results[name]
            assert isinstance(results[name]["score"], float)
            # Mock detector returns ~0.5
            assert 0.0 <= results[name]["score"] <= 1.0

    def test_predict_fallback_uses_default_detector(self) -> None:
        """Simple predict() should use default detector."""
        detector = HybridDetector(
            critical_services=["checkout"],
            critical_detector_type="mock",
            default_detector_type="mock",
        )
        detector.load()

        data = np.random.randn(3, 10).astype(np.float32)
        scores = detector.predict(data)

        assert scores.shape == (3,)
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_predict_raises_if_not_loaded(self) -> None:
        """predict() should raise if load() not called."""
        detector = HybridDetector(critical_services=[])

        with pytest.raises(RuntimeError, match="not loaded"):
            detector.predict(np.random.randn(3, 10).astype(np.float32))


class TestHybridDetectorIntegrationWithStatistical:
    """Integration tests using real statistical detector."""

    def test_hybrid_with_statistical_default(self) -> None:
        """Should work with statistical detector for non-critical metrics."""
        detector = HybridDetector(
            critical_services=["checkout"],
            critical_detector_type="mock",  # Avoid TimesNet dependency
            default_detector_type="statistical",
        )
        detector.load()

        # Normal data (z-scores within threshold)
        data = np.random.randn(2, 100).astype(np.float32) * 0.5  # Low variance
        metric_names = [
            'latency{service="checkout"}',  # → mock
            'latency{service="other"}',  # → statistical
        ]

        results = detector.predict_with_labels(data, metric_names)

        assert results['latency{service="checkout"}']["detector"] == "mock"
        assert results['latency{service="other"}']["detector"] == "statistical"

    def test_statistical_detects_anomaly_in_hybrid(self) -> None:
        """Statistical detector should flag outliers in hybrid mode."""
        detector = HybridDetector(
            critical_services=[],  # All metrics go to statistical
            default_detector_type="statistical",
        )
        detector.load()

        # Create data with an outlier in the last position
        normal = np.zeros((1, 100), dtype=np.float32)
        normal[0, -1] = 10.0  # Big spike at the end

        metric_names = ['cpu{service="worker"}']

        results = detector.predict_with_labels(normal, metric_names)

        # Should have high score due to outlier
        score = results['cpu{service="worker"}']["score"]
        assert score > 0.5  # Definitely anomalous


class TestHybridDetectorWithConfig:
    """Tests for config-based critical services."""

    def test_parses_comma_separated_services(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should parse CRITICAL_SERVICES env var."""
        monkeypatch.setenv("CRITICAL_SERVICES", "checkout,payments,auth")

        # Need to reimport settings to pick up env var
        from anomaly_detection.config import Settings

        test_settings = Settings()
        assert test_settings.critical_services == "checkout,payments,auth"

        # Verify parsing logic works correctly
        services = {
            s.strip()
            for s in test_settings.critical_services.split(",")
            if s.strip()
        }
        assert services == {"checkout", "payments", "auth"}

    def test_handles_empty_critical_services_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should handle empty CRITICAL_SERVICES env var."""
        monkeypatch.setenv("CRITICAL_SERVICES", "")

        from anomaly_detection.config import Settings

        test_settings = Settings()
        assert test_settings.critical_services == ""

        detector = HybridDetector()
        # With empty string, critical_services should be empty set
        assert detector.critical_services == set()



class TestHybridDetectorConfigParsing:
    """Tests for config-based initialization."""

    def test_init_reads_from_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should parse critical_services from settings when not passed as argument."""
        monkeypatch.setenv("CRITICAL_SERVICES", "checkout,payments,auth")

        from anomaly_detection.config import Settings

        with pytest.MonkeyPatch.context() as m:
            m.setattr(
                "anomaly_detection.detector.hybrid.settings",
                Settings(),
            )
            # Don't pass critical_services - should read from settings
            detector = HybridDetector(critical_services=None)
            assert detector.critical_services == {"checkout", "payments", "auth"}

    def test_init_with_spaces_in_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should strip whitespace from service names."""
        monkeypatch.setenv("CRITICAL_SERVICES", " checkout , payments , auth ")

        from anomaly_detection.config import Settings

        with pytest.MonkeyPatch.context() as m:
            m.setattr(
                "anomaly_detection.detector.hybrid.settings",
                Settings(),
            )
            detector = HybridDetector(critical_services=None)
            assert detector.critical_services == {"checkout", "payments", "auth"}


    def test_all_metrics_are_critical(self) -> None:
        """Should handle case where all metrics are critical (no default routing)."""
        detector = HybridDetector(
            critical_services=["checkout", "payments"],
            critical_detector_type="mock",
            default_detector_type="mock",
        )
        detector.load()

        # All metrics belong to critical services
        data = np.random.randn(2, 10).astype(np.float32)
        metric_names = [
            'latency{service="checkout"}',
            'latency{service="payments"}',
        ]

        results = detector.predict_with_labels(data, metric_names)

        assert len(results) == 2
        # All should use critical detector
        for name in metric_names:
            assert results[name]["detector"] == "mock"
