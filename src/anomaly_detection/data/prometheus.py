"""Prometheus/VictoriaMetrics/Gigapipe client."""

from datetime import datetime, timedelta
from typing import Any

import httpx

from anomaly_detection.config import settings
from anomaly_detection.utils.logging import get_logger
from anomaly_detection.utils.retry import retry_async

log = get_logger("prometheus")


class PrometheusClient:
    """Fetch time series data from Prometheus/VictoriaMetrics/Gigapipe."""

    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or settings.prometheus_url).rstrip("/")
        self.client = httpx.AsyncClient(timeout=30.0)

    @retry_async(max_attempts=3, delay_seconds=2.0, exceptions=(httpx.HTTPError,))
    async def query_range(
        self,
        query: str,
        start: datetime,
        end: datetime,
        step: str = "1m",
    ) -> list[dict[str, Any]]:
        """
        Execute a range query.

        Returns list of:
        {
            "metric": {"__name__": "...", "job": "...", ...},
            "values": [[timestamp, "value"], ...]
        }
        """
        params = {
            "query": query,
            "start": start.isoformat() + "Z",
            "end": end.isoformat() + "Z",
            "step": step,
        }

        url = f"{self.base_url}/api/v1/query_range"
        log.debug("prometheus query", url=url, query=query)

        response = await self.client.get(url, params=params)
        response.raise_for_status()

        data = response.json()
        if data["status"] != "success":
            raise ValueError(f"Prometheus error: {data}")

        result: list[dict[str, Any]] = data["data"]["result"]
        return result

    async def fetch_window(
        self,
        query: str,
        window_minutes: int = 96,
        step: str = "1m",
    ) -> list[dict[str, Any]]:
        """Convenience: fetch the last N minutes of data."""
        end = datetime.utcnow()
        start = end - timedelta(minutes=window_minutes)
        return await self.query_range(query, start, end, step)

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()
