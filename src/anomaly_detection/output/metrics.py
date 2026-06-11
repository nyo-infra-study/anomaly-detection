"""Prometheus metrics exporter."""

from prometheus_client import REGISTRY, Gauge, push_to_gateway

from anomaly_detection.config import settings
from anomaly_detection.utils.logging import get_logger

log = get_logger("metrics")

# Define the gauge
anomaly_score_gauge = Gauge(
    "anomaly_score",
    "Anomaly score from TimesNet (0=normal, 1=anomaly)",
    ["metric_name", "severity"],
)


def update_scores(scores: dict[str, float], threshold: float | None = None) -> None:
    """
    Update Prometheus gauge with new scores.

    Args:
        scores: {metric_name: score}
        threshold: score above this is "high" severity
    """
    threshold = threshold or settings.anomaly_threshold

    for metric_name, score in scores.items():
        severity = "high" if score > threshold else "low"
        anomaly_score_gauge.labels(
            metric_name=metric_name,
            severity=severity,
        ).set(score)

    log.debug("metrics updated", count=len(scores))


def push_metrics(job_name: str = "anomaly_detection") -> None:
    """Push metrics to Pushgateway (if configured)."""
    if not settings.pushgateway_url:
        log.debug("pushgateway not configured, skipping push")
        return

    try:
        push_to_gateway(
            settings.pushgateway_url,
            job=job_name,
            registry=REGISTRY,
        )
        log.debug("pushed to pushgateway")
    except Exception as e:
        log.error("pushgateway error", error=str(e))
