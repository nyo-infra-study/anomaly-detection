"""Tests for logging utilities."""

import pytest

from anomaly_detection.utils.logging import get_logger, setup_logging


class TestLogging:
    """Tests for logging module."""

    def test_setup_logging_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should configure JSON logging."""
        monkeypatch.setenv("LOG_FORMAT", "json")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")

        from anomaly_detection.config import Settings

        with pytest.MonkeyPatch.context() as m:
            m.setattr("anomaly_detection.utils.logging.settings", Settings())
            setup_logging()

        # Should not raise
        logger = get_logger("test")
        logger.info("test message")

    def test_setup_logging_console(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should configure console logging."""
        monkeypatch.setenv("LOG_FORMAT", "console")
        monkeypatch.setenv("LOG_LEVEL", "INFO")

        from anomaly_detection.config import Settings

        with pytest.MonkeyPatch.context() as m:
            m.setattr("anomaly_detection.utils.logging.settings", Settings())
            setup_logging()

        logger = get_logger("test")
        logger.info("test message")

    def test_get_logger_returns_bound_logger(self) -> None:
        """Should return a structlog bound logger."""
        logger = get_logger("mylogger")
        # Should have standard logging methods
        assert hasattr(logger, "info")
        assert hasattr(logger, "error")
        assert hasattr(logger, "debug")
        assert hasattr(logger, "warning")

    def test_setup_logging_invalid_level(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should handle invalid log level gracefully."""
        monkeypatch.setenv("LOG_LEVEL", "INVALID_LEVEL")
        monkeypatch.setenv("LOG_FORMAT", "json")

        from anomaly_detection.config import Settings

        with pytest.MonkeyPatch.context() as m:
            m.setattr("anomaly_detection.utils.logging.settings", Settings())
            # Should not raise, falls back to INFO
            setup_logging()

        logger = get_logger("test")
        logger.info("should work")
