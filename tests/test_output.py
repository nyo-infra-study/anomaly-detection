"""Tests for output modules."""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from anomaly_detection.output.grafana import (
    AnomalyAnnotator,
    AnomalyState,
    get_annotator,
)
from anomaly_detection.output.metrics import (
    push_metrics,
    update_scores,
)


class TestMetrics:
    def test_update_scores(self, sample_scores: dict[str, float]) -> None:
        # Should not raise
        update_scores(sample_scores, threshold=0.75)

    def test_update_scores_with_threshold(self) -> None:
        scores = {"metric_a": 0.5}

        # Different thresholds
        update_scores(scores, threshold=0.3)  # high
        update_scores(scores, threshold=0.9)  # low

    def test_update_scores_with_hybrid_format(self) -> None:
        """Should handle hybrid detector dict format."""
        scores = {
            "metric_a": {"score": 0.9, "detector": "timesnet", "service": "checkout"},
            "metric_b": {"score": 0.3, "detector": "statistical", "service": "other"},
        }
        # Should not raise
        update_scores(scores, threshold=0.75)

    def test_update_scores_uses_default_threshold(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should use settings.anomaly_threshold when not specified."""
        monkeypatch.setenv("ANOMALY_THRESHOLD", "0.8")

        from anomaly_detection.config import Settings

        with patch("anomaly_detection.output.metrics.settings", Settings()):
            scores = {"metric_a": 0.85}
            update_scores(scores)  # No threshold specified


class TestPushMetrics:
    def test_push_metrics_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should skip push when pushgateway not configured."""
        monkeypatch.delenv("PUSHGATEWAY_URL", raising=False)

        from anomaly_detection.config import Settings

        with patch("anomaly_detection.output.metrics.settings", Settings()):
            # Should not raise
            push_metrics()

    def test_push_metrics_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should push when pushgateway configured."""
        monkeypatch.setenv("PUSHGATEWAY_URL", "http://pushgateway:9091")

        from anomaly_detection.config import Settings

        mock_push = MagicMock()
        with patch("anomaly_detection.output.metrics.settings", Settings()), patch(
            "anomaly_detection.output.metrics.push_to_gateway", mock_push
        ):
            push_metrics(job_name="test_job")
            mock_push.assert_called_once()

    def test_push_metrics_handles_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should log error when push fails."""
        monkeypatch.setenv("PUSHGATEWAY_URL", "http://pushgateway:9091")

        from anomaly_detection.config import Settings

        mock_push = MagicMock(side_effect=Exception("connection failed"))
        with patch("anomaly_detection.output.metrics.settings", Settings()), patch(
            "anomaly_detection.output.metrics.push_to_gateway", mock_push
        ):
            # Should not raise
            push_metrics()


class TestAnomalyState:
    def test_default_values(self) -> None:
        state = AnomalyState()
        assert state.is_anomalous is False
        assert state.started_at is None
        assert state.max_score == 0.0


class TestAnomalyAnnotator:
    @pytest.mark.asyncio
    async def test_disabled_without_config(self) -> None:
        """Annotator should gracefully handle missing config."""
        annotator = AnomalyAnnotator(
            grafana_url=None,
            api_token=None,
        )
        annotator.states = {}

        # Should not raise
        await annotator.update("test_metric", 0.9)
        await annotator.close()

    @pytest.mark.asyncio
    async def test_state_tracking_start(self) -> None:
        """Verify anomaly start is tracked."""
        annotator = AnomalyAnnotator(
            grafana_url="http://fake",
            api_token="fake",
        )
        annotator.states = {}
        annotator.enabled = False  # Disable actual HTTP calls

        now = datetime.utcnow()

        # Normal → nothing happens
        await annotator.update("m1", 0.3, threshold=0.75, timestamp=now)
        assert (
            annotator.states.get("m1") is None
            or not annotator.states["m1"].is_anomalous
        )

        # Spike → anomaly starts
        await annotator.update(
            "m1", 0.9, threshold=0.75, timestamp=now + timedelta(minutes=1)
        )
        assert annotator.states["m1"].is_anomalous
        assert annotator.states["m1"].started_at is not None
        assert annotator.states["m1"].max_score == 0.9

        await annotator.close()

    @pytest.mark.asyncio
    async def test_state_tracking_end(self) -> None:
        """Verify anomaly end resets state."""
        annotator = AnomalyAnnotator(
            grafana_url="http://fake",
            api_token="fake",
        )
        annotator.states = {}
        annotator.enabled = False

        now = datetime.utcnow()

        # Start anomaly
        await annotator.update("m1", 0.9, threshold=0.75, timestamp=now)
        assert annotator.states["m1"].is_anomalous

        # End anomaly
        await annotator.update(
            "m1", 0.3, threshold=0.75, timestamp=now + timedelta(minutes=1)
        )
        assert not annotator.states["m1"].is_anomalous
        assert annotator.states["m1"].started_at is None

        await annotator.close()

    @pytest.mark.asyncio
    async def test_max_score_tracking(self) -> None:
        """Verify max score is tracked during anomaly."""
        annotator = AnomalyAnnotator(
            grafana_url="http://fake",
            api_token="fake",
        )
        annotator.states = {}
        annotator.enabled = False

        now = datetime.utcnow()

        # Start at 0.8
        await annotator.update("m1", 0.8, threshold=0.75, timestamp=now)
        assert annotator.states["m1"].max_score == 0.8

        # Peak at 0.95
        await annotator.update(
            "m1", 0.95, threshold=0.75, timestamp=now + timedelta(minutes=1)
        )
        assert annotator.states["m1"].max_score == 0.95

        # Drop but still anomalous
        await annotator.update(
            "m1", 0.85, threshold=0.75, timestamp=now + timedelta(minutes=2)
        )
        assert annotator.states["m1"].max_score == 0.95  # Still the peak

        await annotator.close()

    @pytest.mark.asyncio
    async def test_write_annotation_success(self) -> None:
        """Should write annotation to Grafana API."""
        annotator = AnomalyAnnotator(
            grafana_url="http://grafana:3000",
            api_token="test-token",
            dashboard_uid="abc123",
        )
        annotator.states = {}
        annotator.enabled = True

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        annotator.client.post = AsyncMock(return_value=mock_response)

        now = datetime.utcnow()

        # Start anomaly
        await annotator.update("m1", 0.9, threshold=0.75, timestamp=now)

        # End anomaly → should trigger annotation
        await annotator.update(
            "m1", 0.3, threshold=0.75, timestamp=now + timedelta(minutes=5)
        )

        # Verify POST was called
        annotator.client.post.assert_called_once()
        call_args = annotator.client.post.call_args
        assert call_args[0][0] == "http://grafana:3000/api/annotations"
        assert "dashboardUID" in call_args[1]["json"]

        await annotator.close()

    @pytest.mark.asyncio
    async def test_write_annotation_error(self) -> None:
        """Should handle annotation API errors gracefully."""
        annotator = AnomalyAnnotator(
            grafana_url="http://grafana:3000",
            api_token="test-token",
        )
        annotator.states = {}
        annotator.enabled = True

        annotator.client.post = AsyncMock(side_effect=Exception("API error"))

        now = datetime.utcnow()

        # Start and end anomaly
        await annotator.update("m1", 0.9, threshold=0.75, timestamp=now)
        # Should not raise
        await annotator.update(
            "m1", 0.3, threshold=0.75, timestamp=now + timedelta(minutes=5)
        )

        await annotator.close()

    @pytest.mark.asyncio
    async def test_uses_default_threshold(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should use settings.anomaly_threshold when not specified."""
        monkeypatch.setenv("ANOMALY_THRESHOLD", "0.5")

        from anomaly_detection.config import Settings

        annotator = AnomalyAnnotator(
            grafana_url="http://fake",
            api_token="fake",
        )
        annotator.states = {}
        annotator.enabled = False

        with patch("anomaly_detection.output.grafana.settings", Settings()):
            now = datetime.utcnow()
            # 0.6 > 0.5 threshold → should be anomaly
            await annotator.update("m1", 0.6, timestamp=now)
            assert annotator.states["m1"].is_anomalous

        await annotator.close()


class TestGetAnnotator:
    def test_get_annotator_singleton(self) -> None:
        """Should return the same instance."""
        import anomaly_detection.output.grafana as grafana_module

        grafana_module._annotator = None

        a1 = get_annotator()
        a2 = get_annotator()

        assert a1 is a2

        grafana_module._annotator = None

    def test_get_annotator_initializes_states(self) -> None:
        """Should initialize states dict."""
        import anomaly_detection.output.grafana as grafana_module

        grafana_module._annotator = None

        annotator = get_annotator()
        assert annotator.states == {}

        grafana_module._annotator = None
