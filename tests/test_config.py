"""Tests for configuration."""

import pytest


class TestConfig:
    def test_default_values(self) -> None:
        """Config has sensible defaults."""
        from anomaly_detection.config import Settings

        settings = Settings()

        assert settings.prometheus_url == "http://localhost:9090"
        assert settings.window_size == 96
        assert settings.anomaly_threshold == 0.75
        assert settings.log_level == "INFO"
        assert settings.log_format == "json"

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Environment variables override defaults."""
        monkeypatch.setenv("PROMETHEUS_URL", "http://custom:9090")
        monkeypatch.setenv("ANOMALY_THRESHOLD", "0.8")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")

        from anomaly_detection.config import Settings

        settings = Settings()

        assert settings.prometheus_url == "http://custom:9090"
        assert settings.anomaly_threshold == 0.8
        assert settings.log_level == "DEBUG"

    def test_optional_values(self) -> None:
        """Optional values default to None."""
        from anomaly_detection.config import Settings

        settings = Settings()

        assert settings.pushgateway_url is None
        assert settings.grafana_url is None
        assert settings.grafana_api_token is None
