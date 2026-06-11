"""Grafana annotation writer for Datadog-style shaded regions."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import httpx

from anomaly_detection.config import settings
from anomaly_detection.utils.logging import get_logger

log = get_logger("grafana")


@dataclass
class AnomalyState:
    """Track anomaly start/end for one metric."""

    is_anomalous: bool = False
    started_at: datetime | None = None
    max_score: float = 0.0


class AnomalyAnnotator:
    """
    Writes Grafana annotations when anomalies end.
    Creates shaded regions showing anomaly duration (Datadog-style).
    """

    def __init__(
        self,
        grafana_url: str | None = None,
        api_token: str | None = None,
        dashboard_uid: str | None = None,
    ):
        self.grafana_url = (grafana_url or settings.grafana_url or "").rstrip("/")
        self.api_token = api_token or settings.grafana_api_token
        self.dashboard_uid = dashboard_uid or settings.grafana_dashboard_uid

        self.enabled = bool(self.grafana_url and self.api_token)
        if not self.enabled:
            log.warning("grafana annotations disabled (missing url or token)")

        self.client = httpx.AsyncClient(
            timeout=10.0,
            headers=(
                {
                    "Authorization": f"Bearer {self.api_token}",
                    "Content-Type": "application/json",
                }
                if self.api_token
                else {}
            ),
        )

        # Track state per metric
        self.states: dict[str, AnomalyState] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.states = {}

    async def update(
        self,
        metric_name: str,
        score: float,
        threshold: float | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        """
        Call this every inference cycle.
        When anomaly ends, writes annotation to Grafana.
        """
        threshold = threshold or settings.anomaly_threshold
        timestamp = timestamp or datetime.utcnow()
        is_anomalous = score > threshold

        # Get or create state
        if metric_name not in self.states:
            self.states[metric_name] = AnomalyState()
        state = self.states[metric_name]

        # ── Anomaly just started ──
        if is_anomalous and not state.is_anomalous:
            state.is_anomalous = True
            state.started_at = timestamp
            state.max_score = score
            log.info("anomaly started", metric=metric_name, score=score)

        # ── Anomaly ongoing — track max score ──
        elif is_anomalous and state.is_anomalous:
            state.max_score = max(state.max_score, score)

        # ── Anomaly just ended → write annotation ──
        elif not is_anomalous and state.is_anomalous:
            if state.started_at is not None and self.enabled:
                await self._write_annotation(
                    metric_name=metric_name,
                    start_time=state.started_at,
                    end_time=timestamp,
                    max_score=state.max_score,
                )
            state.is_anomalous = False
            state.started_at = None
            state.max_score = 0.0

    async def _write_annotation(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: datetime,
        max_score: float,
    ) -> None:
        """Write a time-range annotation to Grafana API."""
        payload: dict[str, Any] = {
            "time": int(start_time.timestamp() * 1000),
            "timeEnd": int(end_time.timestamp() * 1000),
            "tags": ["anomaly", metric_name],
            "text": f"Anomaly on {metric_name} (max score: {max_score:.2f})",
        }

        if self.dashboard_uid:
            payload["dashboardUID"] = self.dashboard_uid

        try:
            response = await self.client.post(
                f"{self.grafana_url}/api/annotations",
                json=payload,
            )
            response.raise_for_status()

            log.info(
                "annotation created",
                metric=metric_name,
                duration_sec=(end_time - start_time).total_seconds(),
                max_score=max_score,
            )
        except Exception as e:
            log.error(
                "annotation failed",
                metric=metric_name,
                error=str(e),
            )

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()


# Singleton for convenience
_annotator: AnomalyAnnotator | None = None


def get_annotator() -> AnomalyAnnotator:
    """Get or create the global annotator instance."""
    global _annotator
    if _annotator is None:
        _annotator = AnomalyAnnotator()
        _annotator.states = {}  # Initialize states dict
    return _annotator
