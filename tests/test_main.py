"""Tests for main module."""

import asyncio
import signal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from anomaly_detection.main import (
    _extract_score,
    _wait_or_shutdown,
    handle_shutdown,
    main,
    run_loop,
)


class TestExtractScore:
    """Tests for _extract_score helper."""

    def test_extract_from_float(self) -> None:
        """Should return float directly."""
        assert _extract_score(0.75) == 0.75

    def test_extract_from_dict(self) -> None:
        """Should extract score from dict."""
        value = {"score": 0.85, "detector": "timesnet", "service": "checkout"}
        assert _extract_score(value) == 0.85


class TestHandleShutdown:
    """Tests for shutdown handler."""

    def test_handle_shutdown_sets_event(self) -> None:
        """Should set the shutdown event."""
        import anomaly_detection.main as main_module

        event = asyncio.Event()
        main_module._shutdown_event = event

        handle_shutdown(signal.SIGTERM)

        assert event.is_set()
        main_module._shutdown_event = None

    def test_handle_shutdown_no_event(self) -> None:
        """Should not raise if no event set."""
        import anomaly_detection.main as main_module

        main_module._shutdown_event = None
        # Should not raise
        handle_shutdown(signal.SIGINT)


class TestWaitOrShutdown:
    """Tests for _wait_or_shutdown helper."""

    @pytest.mark.asyncio
    async def test_wait_timeout(self) -> None:
        """Should wait for specified time when no shutdown."""
        import anomaly_detection.main as main_module

        event = asyncio.Event()
        main_module._shutdown_event = event

        import time

        start = time.time()
        await _wait_or_shutdown(0.1)
        elapsed = time.time() - start

        assert elapsed >= 0.1
        main_module._shutdown_event = None

    @pytest.mark.asyncio
    async def test_wait_shutdown(self) -> None:
        """Should return early when shutdown signaled."""
        import anomaly_detection.main as main_module

        event = asyncio.Event()
        main_module._shutdown_event = event

        # Set event after a short delay
        async def set_event():
            await asyncio.sleep(0.05)
            event.set()

        asyncio.create_task(set_event())

        import time

        start = time.time()
        await _wait_or_shutdown(10.0)  # Would wait 10s without shutdown
        elapsed = time.time() - start

        assert elapsed < 1.0  # Should have returned early
        main_module._shutdown_event = None

    @pytest.mark.asyncio
    async def test_wait_no_event(self) -> None:
        """Should sleep normally if no event."""
        import anomaly_detection.main as main_module

        main_module._shutdown_event = None

        import time

        start = time.time()
        await _wait_or_shutdown(0.1)
        elapsed = time.time() - start

        assert elapsed >= 0.1


class TestRunLoop:
    """Tests for the main run loop."""

    @pytest.mark.asyncio
    async def test_run_loop_single_cycle(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should run one inference cycle then shutdown."""
        # Mock all components
        mock_prom = AsyncMock()
        mock_prom.fetch_window = AsyncMock(
            return_value=[
                {
                    "metric": {"__name__": "test"},
                    "values": [[i, str(float(i))] for i in range(100)],
                }
            ]
        )
        mock_prom.close = AsyncMock()

        mock_detector = MagicMock()
        mock_detector.load = MagicMock()
        mock_detector.predict_with_labels = MagicMock(return_value={"test": 0.5})

        mock_annotator = AsyncMock()
        mock_annotator.update = AsyncMock()
        mock_annotator.close = AsyncMock()

        mock_health_runner = AsyncMock()
        mock_health_runner.cleanup = AsyncMock()

        with patch(
            "anomaly_detection.main.PrometheusClient", return_value=mock_prom
        ), patch(
            "anomaly_detection.main.get_detector", return_value=mock_detector
        ), patch(
            "anomaly_detection.main.get_annotator", return_value=mock_annotator
        ), patch(
            "anomaly_detection.main.start_health_server",
            return_value=mock_health_runner,
        ), patch(
            "anomaly_detection.main.push_metrics"
        ), patch(
            "anomaly_detection.main.update_scores"
        ), patch(
            "anomaly_detection.main.set_healthy"
        ), patch(
            "anomaly_detection.main.record_successful_cycle"
        ):
            # Trigger shutdown after first cycle
            import anomaly_detection.main as main_module

            async def shutdown_after_delay():
                await asyncio.sleep(0.1)
                if main_module._shutdown_event:
                    main_module._shutdown_event.set()

            shutdown_task = asyncio.create_task(shutdown_after_delay())
            _ = shutdown_task  # Keep reference to avoid warning

            # Patch settings for fast cycle
            monkeypatch.setenv("FETCH_INTERVAL_SECONDS", "1")

            await run_loop()

            # Verify components were called
            mock_detector.load.assert_called_once()
            mock_prom.close.assert_called()
            mock_annotator.close.assert_called()

    @pytest.mark.asyncio
    async def test_run_loop_handles_empty_data(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should handle empty data gracefully."""
        mock_prom = AsyncMock()
        mock_prom.fetch_window = AsyncMock(return_value=[])
        mock_prom.close = AsyncMock()

        mock_detector = MagicMock()
        mock_detector.load = MagicMock()

        mock_annotator = AsyncMock()
        mock_annotator.close = AsyncMock()

        mock_health_runner = AsyncMock()
        mock_health_runner.cleanup = AsyncMock()

        with patch(
            "anomaly_detection.main.PrometheusClient", return_value=mock_prom
        ), patch(
            "anomaly_detection.main.get_detector", return_value=mock_detector
        ), patch(
            "anomaly_detection.main.get_annotator", return_value=mock_annotator
        ), patch(
            "anomaly_detection.main.start_health_server",
            return_value=mock_health_runner,
        ), patch(
            "anomaly_detection.main.set_healthy"
        ):
            import anomaly_detection.main as main_module

            async def shutdown_after_delay():
                await asyncio.sleep(0.1)
                if main_module._shutdown_event:
                    main_module._shutdown_event.set()

            asyncio.create_task(shutdown_after_delay())

            await run_loop()

            # Should not crash with empty data

    @pytest.mark.asyncio
    async def test_run_loop_handles_cycle_exception(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should catch and log cycle exceptions."""
        mock_prom = AsyncMock()
        mock_prom.fetch_window = AsyncMock(side_effect=ValueError("fetch failed"))
        mock_prom.close = AsyncMock()

        mock_detector = MagicMock()
        mock_detector.load = MagicMock()

        mock_annotator = AsyncMock()
        mock_annotator.close = AsyncMock()

        mock_health_runner = AsyncMock()
        mock_health_runner.cleanup = AsyncMock()

        mock_set_healthy = MagicMock()

        with patch(
            "anomaly_detection.main.PrometheusClient", return_value=mock_prom
        ), patch(
            "anomaly_detection.main.get_detector", return_value=mock_detector
        ), patch(
            "anomaly_detection.main.get_annotator", return_value=mock_annotator
        ), patch(
            "anomaly_detection.main.start_health_server",
            return_value=mock_health_runner,
        ), patch(
            "anomaly_detection.main.set_healthy", mock_set_healthy
        ):
            import anomaly_detection.main as main_module

            async def shutdown_after_delay():
                await asyncio.sleep(0.1)
                if main_module._shutdown_event:
                    main_module._shutdown_event.set()

            asyncio.create_task(shutdown_after_delay())

            await run_loop()

            # Should have set healthy to False on error
            # (Last call in finally is False)
            assert mock_set_healthy.call_count >= 1


class TestMain:
    """Tests for main entry point."""

    def test_main_runs_loop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should setup logging and run the loop."""
        mock_setup = MagicMock()

        # Create a mock that properly consumes the coroutine
        def mock_asyncio_run(coro: Any) -> None:
            # Close the coroutine to avoid warning
            coro.close()

        with patch("anomaly_detection.main.setup_logging", mock_setup), patch(
            "anomaly_detection.main.asyncio.run", mock_asyncio_run
        ):
            main()

            mock_setup.assert_called_once()
