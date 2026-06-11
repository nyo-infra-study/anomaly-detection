"""Tests for health check server."""

import time

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from anomaly_detection import health


class TestHealthFunctions:
    """Tests for health module functions."""

    def test_set_healthy_true(self) -> None:
        """Should set healthy state to True."""
        health._healthy = False
        health.set_healthy(True)
        assert health._healthy is True

    def test_set_healthy_false(self) -> None:
        """Should set healthy state to False."""
        health._healthy = True
        health.set_healthy(False)
        assert health._healthy is False

    def test_record_successful_cycle(self) -> None:
        """Should record current time."""
        health._last_successful_cycle = 0
        before = time.time()
        health.record_successful_cycle()
        after = time.time()

        assert before <= health._last_successful_cycle <= after


class TestHealthHandlersIntegration:
    """Integration tests using aiohttp test client."""

    @pytest.mark.asyncio
    async def test_health_endpoint_healthy(self) -> None:
        """Should return 200 when healthy."""
        health._healthy = True

        app = web.Application()
        app.router.add_get("/health", health.health_handler)

        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/health")
            assert resp.status == 200
            data = await resp.json()
            assert data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_endpoint_unhealthy(self) -> None:
        """Should return 503 when unhealthy."""
        health._healthy = False

        app = web.Application()
        app.router.add_get("/health", health.health_handler)

        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/health")
            assert resp.status == 503
            data = await resp.json()
            assert data["status"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_ready_endpoint_ready(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should return 200 when ready (recent successful cycle)."""
        monkeypatch.setenv("FETCH_INTERVAL_SECONDS", "60")
        health._last_successful_cycle = time.time()

        app = web.Application()
        app.router.add_get("/ready", health.ready_handler)

        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/ready")
            assert resp.status == 200
            data = await resp.json()
            assert data["status"] == "ready"

    @pytest.mark.asyncio
    async def test_ready_endpoint_not_ready_no_cycle(self) -> None:
        """Should return 503 when no successful cycle yet."""
        health._last_successful_cycle = 0

        app = web.Application()
        app.router.add_get("/ready", health.ready_handler)

        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/ready")
            assert resp.status == 503
            data = await resp.json()
            assert data["status"] == "not ready"

    @pytest.mark.asyncio
    async def test_ready_endpoint_not_ready_stale(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should return 503 when last cycle is too old."""
        monkeypatch.setenv("FETCH_INTERVAL_SECONDS", "60")
        from anomaly_detection.config import Settings

        with pytest.MonkeyPatch.context() as m:
            m.setattr("anomaly_detection.health.settings", Settings())
            # Set last successful cycle to 10 minutes ago (stale)
            health._last_successful_cycle = time.time() - 600

            app = web.Application()
            app.router.add_get("/ready", health.ready_handler)

            async with TestClient(TestServer(app)) as client:
                resp = await client.get("/ready")
                assert resp.status == 503


class TestHealthServer:
    """Tests for health server startup."""

    @pytest.mark.asyncio
    async def test_start_health_server(self) -> None:
        """Should start and return a runner."""
        runner = await health.start_health_server(port=18080)
        assert runner is not None
        await runner.cleanup()
