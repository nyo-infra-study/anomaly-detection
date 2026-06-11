"""Tests for Prometheus client."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from anomaly_detection.data.prometheus import PrometheusClient


class TestPrometheusClient:
    """Tests for PrometheusClient."""

    def test_init_with_default_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should use settings.prometheus_url by default."""
        monkeypatch.setenv("PROMETHEUS_URL", "http://prom:9090")
        from anomaly_detection.config import Settings

        with patch("anomaly_detection.data.prometheus.settings", Settings()):
            client = PrometheusClient()
            assert client.base_url == "http://prom:9090"

    def test_init_with_custom_url(self) -> None:
        """Should accept custom URL."""
        client = PrometheusClient(base_url="http://custom:9090/")
        assert client.base_url == "http://custom:9090"  # Trailing slash stripped

    @pytest.mark.asyncio
    async def test_query_range_success(self) -> None:
        """Should parse successful Prometheus response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "success",
            "data": {
                "result": [
                    {
                        "metric": {"__name__": "up", "job": "test"},
                        "values": [[1000, "1"], [1060, "1"]],
                    }
                ]
            },
        }
        mock_response.raise_for_status = MagicMock()

        client = PrometheusClient(base_url="http://test:9090")
        client.client.get = AsyncMock(return_value=mock_response)

        result = await client.query_range(
            query="up",
            start=datetime(2024, 1, 1, 0, 0),
            end=datetime(2024, 1, 1, 1, 0),
            step="1m",
        )

        assert len(result) == 1
        assert result[0]["metric"]["__name__"] == "up"
        await client.close()

    @pytest.mark.asyncio
    async def test_query_range_error_status(self) -> None:
        """Should raise on Prometheus error status."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "error",
            "errorType": "bad_data",
            "error": "invalid query",
        }
        mock_response.raise_for_status = MagicMock()

        client = PrometheusClient(base_url="http://test:9090")
        client.client.get = AsyncMock(return_value=mock_response)

        with pytest.raises(ValueError, match="Prometheus error"):
            await client.query_range(
                query="invalid{",
                start=datetime(2024, 1, 1),
                end=datetime(2024, 1, 1, 1, 0),
            )
        await client.close()

    @pytest.mark.asyncio
    async def test_query_range_http_error(self) -> None:
        """Should propagate HTTP errors after retries."""
        client = PrometheusClient(base_url="http://test:9090")
        client.client.get = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "Server error",
                request=MagicMock(),
                response=MagicMock(status_code=500),
            )
        )

        with pytest.raises(httpx.HTTPStatusError):
            await client.query_range(
                query="up",
                start=datetime(2024, 1, 1),
                end=datetime(2024, 1, 1, 1, 0),
            )
        await client.close()

    @pytest.mark.asyncio
    async def test_fetch_window(self) -> None:
        """Should call query_range with computed time range."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "success",
            "data": {"result": []},
        }
        mock_response.raise_for_status = MagicMock()

        client = PrometheusClient(base_url="http://test:9090")
        client.client.get = AsyncMock(return_value=mock_response)

        result = await client.fetch_window(query="up", window_minutes=60)

        assert result == []
        # Verify the call was made
        client.client.get.assert_called_once()
        await client.close()

    @pytest.mark.asyncio
    async def test_close(self) -> None:
        """Should close the HTTP client."""
        client = PrometheusClient(base_url="http://test:9090")
        client.client.aclose = AsyncMock()

        await client.close()

        client.client.aclose.assert_called_once()
