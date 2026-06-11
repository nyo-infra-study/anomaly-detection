"""Tests for output modules."""

from datetime import datetime, timedelta

import pytest

from anomaly_detection.output.grafana import AnomalyAnnotator, AnomalyState
from anomaly_detection.output.metrics import update_scores


class TestMetrics:
    def test_update_scores(self, sample_scores: dict[str, float]) -> None:
        # Should not raise
        update_scores(sample_scores, threshold=0.75)

    def test_update_scores_with_threshold(self) -> None:
        scores = {"metric_a": 0.5}

        # Different thresholds
        update_scores(scores, threshold=0.3)  # high
        update_scores(scores, threshold=0.9)  # low


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
        assert annotator.states.get("m1") is None or not annotator.states["m1"].is_anomalous

        # Spike → anomaly starts
        await annotator.update("m1", 0.9, threshold=0.75, timestamp=now + timedelta(minutes=1))
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
        await annotator.update("m1", 0.3, threshold=0.75, timestamp=now + timedelta(minutes=1))
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
        await annotator.update("m1", 0.95, threshold=0.75, timestamp=now + timedelta(minutes=1))
        assert annotator.states["m1"].max_score == 0.95

        # Drop but still anomalous
        await annotator.update("m1", 0.85, threshold=0.75, timestamp=now + timedelta(minutes=2))
        assert annotator.states["m1"].max_score == 0.95  # Still the peak

        await annotator.close()
