"""Health check HTTP server for Kubernetes probes."""

import time

from aiohttp import web

from anomaly_detection.config import settings
from anomaly_detection.utils.logging import get_logger

log = get_logger("health")

# Global state for health checks
_healthy = True
_last_successful_cycle: float = 0


def set_healthy(healthy: bool) -> None:
    """Set the health status."""
    global _healthy
    _healthy = healthy


def record_successful_cycle() -> None:
    """Record a successful inference cycle."""
    global _last_successful_cycle
    _last_successful_cycle = time.time()


async def health_handler(request: web.Request) -> web.Response:
    """Liveness probe: is the service running?"""
    if _healthy:
        return web.json_response({"status": "healthy"})
    return web.json_response({"status": "unhealthy"}, status=503)


async def ready_handler(request: web.Request) -> web.Response:
    """Readiness probe: has completed at least one cycle recently."""
    current_time = time.time()
    max_age = settings.fetch_interval_seconds * 5  # Allow 5 missed cycles

    if _last_successful_cycle > 0 and (current_time - _last_successful_cycle) < max_age:
        return web.json_response({"status": "ready"})
    return web.json_response({"status": "not ready"}, status=503)


async def start_health_server(port: int | None = None) -> web.AppRunner:
    """Start health check HTTP server."""
    port = port or settings.health_port

    app = web.Application()
    app.router.add_get("/health", health_handler)
    app.router.add_get("/ready", ready_handler)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()

    log.info("health server started", port=port)
    return runner
