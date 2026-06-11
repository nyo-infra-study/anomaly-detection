"""Main entry point for the anomaly detection service."""

import asyncio
import signal
from typing import Any

from anomaly_detection.config import settings
from anomaly_detection.data.prometheus import PrometheusClient
from anomaly_detection.detector import get_detector
from anomaly_detection.detector.preprocessor import (
    normalize,
    pad_or_truncate,
    prometheus_to_array,
)
from anomaly_detection.health import (
    record_successful_cycle,
    set_healthy,
    start_health_server,
)
from anomaly_detection.output.grafana import get_annotator
from anomaly_detection.output.metrics import push_metrics, update_scores
from anomaly_detection.utils.logging import get_logger, setup_logging

log = get_logger("main")

# Shutdown flag
_shutdown_event: asyncio.Event | None = None


def handle_shutdown(sig: signal.Signals) -> None:
    """Handle shutdown signals."""
    log.info("received shutdown signal", signal=sig.name)
    if _shutdown_event:
        _shutdown_event.set()


def _extract_score(value: float | dict[str, Any]) -> float:
    """Extract score from either simple float or hybrid dict format."""
    if isinstance(value, dict):
        return value["score"]
    return value


async def run_loop() -> None:
    """Main inference loop."""
    global _shutdown_event
    _shutdown_event = asyncio.Event()

    # Register signal handlers
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, handle_shutdown, sig)

    # Start health server
    health_runner = await start_health_server()

    # Initialize components
    prom_client = PrometheusClient()
    detector = get_detector()
    detector.load()
    annotator = get_annotator()

    set_healthy(True)

    try:
        while not _shutdown_event.is_set():
            try:
                log.info("cycle start")

                # 1. Fetch
                result = await prom_client.fetch_window(
                    query=settings.prometheus_query,
                    window_minutes=settings.window_size,
                )

                # 2. Preprocess
                data, metric_names = prometheus_to_array(result)
                if data.size == 0:
                    log.warning("no data returned")
                    await _wait_or_shutdown(settings.fetch_interval_seconds)
                    continue

                data = pad_or_truncate(data, settings.window_size)
                data_norm, means, stds = normalize(data)

                # 3. Inference
                # predict_with_labels returns either:
                #   - dict[str, float] for simple detectors
                #   - dict[str, dict] for hybrid detector (with score, detector, service keys)
                scores = detector.predict_with_labels(data_norm, metric_names)

                # 4. Output
                update_scores(scores)
                push_metrics()

                for metric_name, value in scores.items():
                    score = _extract_score(value)
                    await annotator.update(metric_name, score)

                # Mark cycle complete
                record_successful_cycle()

                high_count = sum(
                    1
                    for v in scores.values()
                    if _extract_score(v) > settings.anomaly_threshold
                )
                log.info("cycle complete", total=len(scores), anomalies=high_count)

            except Exception as e:
                log.error("cycle failed", error=str(e))
                set_healthy(False)

            # Wait for next cycle or shutdown
            await _wait_or_shutdown(settings.fetch_interval_seconds)

    finally:
        log.info("cleaning up")
        set_healthy(False)
        await prom_client.close()
        await annotator.close()
        await health_runner.cleanup()


async def _wait_or_shutdown(seconds: float) -> None:
    """Wait for specified time or until shutdown signal."""
    if _shutdown_event is None:
        await asyncio.sleep(seconds)
        return

    try:
        await asyncio.wait_for(_shutdown_event.wait(), timeout=seconds)
    except TimeoutError:
        pass  # Normal — continue to next cycle


def main() -> None:
    """Entry point."""
    setup_logging()
    log.info(
        "anomaly-detection starting",
        version="0.1.0",
        prometheus_url=settings.prometheus_url,
        query=settings.prometheus_query,
    )

    asyncio.run(run_loop())
    log.info("anomaly-detection stopped")


if __name__ == "__main__":
    main()
