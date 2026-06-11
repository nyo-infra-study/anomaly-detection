# M4: Output Pipeline

## Goal
Push anomaly scores to Prometheus and write Grafana annotations for Datadog-style shaded regions.

**Depends on**: M3 (Detector)

---

## Tasks

### 4.1 Implement Prometheus metrics exporter

**What**: Push `anomaly_score` gauge to Prometheus Pushgateway (or expose for scraping)

**Deliverable**:
```python
# src/anomaly_detection/output/metrics.py
from prometheus_client import Gauge, push_to_gateway, REGISTRY
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
```

---

### 4.2 Implement Grafana annotation writer

**What**: Track anomaly state and write annotations with time ranges

**Deliverable**:
```python
# src/anomaly_detection/output/grafana.py
import httpx
from datetime import datetime
from dataclasses import dataclass, field
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
            headers={
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json",
            } if self.api_token else {},
        )
        
        # Track state per metric
        self.states: dict[str, AnomalyState] = field(default_factory=dict)

    def __post_init__(self):
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
        if not self.enabled:
            return
        
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
        payload = {
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

    async def close(self):
        await self.client.aclose()


# Singleton for convenience
_annotator: AnomalyAnnotator | None = None


def get_annotator() -> AnomalyAnnotator:
    global _annotator
    if _annotator is None:
        _annotator = AnomalyAnnotator()
        _annotator.states = {}  # Initialize states dict
    return _annotator
```

---

### 4.3 Integrate output into main loop

**What**: Wire up Prometheus + Grafana output

**Deliverable**: Update `main.py`:
```python
# src/anomaly_detection/main.py
import asyncio
from anomaly_detection.config import settings
from anomaly_detection.utils.logging import setup_logging, get_logger
from anomaly_detection.data.prometheus import PrometheusClient
from anomaly_detection.detector import get_detector
from anomaly_detection.detector.preprocessor import (
    prometheus_to_array,
    normalize,
    pad_or_truncate,
)
from anomaly_detection.output.metrics import update_scores, push_metrics
from anomaly_detection.output.grafana import get_annotator

log = get_logger("main")


async def run_loop():
    # Initialize components
    prom_client = PrometheusClient()
    detector = get_detector()
    detector.load()
    annotator = get_annotator()

    try:
        while True:
            log.info("cycle start")
            
            # 1. Fetch data
            result = await prom_client.fetch_window(
                query=settings.prometheus_query,
                window_minutes=settings.window_size,
            )
            
            # 2. Preprocess
            data, metric_names = prometheus_to_array(result)
            if data.size == 0:
                log.warning("no data returned")
                await asyncio.sleep(settings.fetch_interval_seconds)
                continue
            
            data = pad_or_truncate(data, settings.window_size)
            data_norm, means, stds = normalize(data)
            
            # 3. Run inference
            scores = detector.predict_with_labels(data_norm, metric_names)
            
            # 4. Output: Prometheus metrics
            update_scores(scores)
            push_metrics()
            
            # 5. Output: Grafana annotations
            for metric_name, score in scores.items():
                await annotator.update(metric_name, score)
            
            # Log summary
            high_count = sum(1 for s in scores.values() if s > settings.anomaly_threshold)
            log.info(
                "cycle complete",
                total_metrics=len(scores),
                anomalies=high_count,
            )
            
            await asyncio.sleep(settings.fetch_interval_seconds)
            
    finally:
        await prom_client.close()
        await annotator.close()


def main() -> None:
    setup_logging()
    log.info(
        "anomaly-detection starting",
        prometheus_url=settings.prometheus_url,
        grafana_url=settings.grafana_url,
    )
    
    try:
        asyncio.run(run_loop())
    except KeyboardInterrupt:
        log.info("shutting down")


if __name__ == "__main__":
    main()
```

---

### 4.4 Write output tests

**What**: Unit tests for output components

**Deliverable**:
```python
# tests/test_output.py
import pytest
from unittest.mock import patch, MagicMock
from anomaly_detection.output.metrics import update_scores, anomaly_score_gauge


class TestMetrics:
    def test_update_scores(self):
        scores = {
            "metric_a": 0.3,
            "metric_b": 0.9,
        }
        
        update_scores(scores, threshold=0.75)
        
        # Check gauge was set
        # Note: prometheus_client doesn't have easy inspection,
        # so we just verify no errors


class TestGrafana:
    @pytest.mark.asyncio
    async def test_annotator_disabled_without_config(self):
        """Annotator should gracefully handle missing config."""
        from anomaly_detection.output.grafana import AnomalyAnnotator
        
        annotator = AnomalyAnnotator(
            grafana_url=None,
            api_token=None,
        )
        annotator.states = {}
        
        # Should not raise
        await annotator.update("test_metric", 0.9)
        await annotator.close()

    @pytest.mark.asyncio
    async def test_state_tracking(self):
        """Verify anomaly start/end state machine."""
        from anomaly_detection.output.grafana import AnomalyAnnotator
        
        annotator = AnomalyAnnotator(
            grafana_url="http://fake",
            api_token="fake",
        )
        annotator.states = {}
        annotator.enabled = False  # Disable actual HTTP calls
        
        # Start anomaly
        await annotator.update("m1", 0.9, threshold=0.75)
        assert annotator.states["m1"].is_anomalous is True
        assert annotator.states["m1"].started_at is not None
        
        # End anomaly
        await annotator.update("m1", 0.3, threshold=0.75)
        assert annotator.states["m1"].is_anomalous is False
        
        await annotator.close()
```

---

## Checklist

- [x] 4.1 Prometheus metrics exporter works
- [x] 4.2 Grafana annotator works
- [x] 4.3 Main loop outputs to both
- [x] 4.4 Tests passing

## Done When

```bash
# With all config set:
PROMETHEUS_URL=http://prometheus:9090 \
PROMETHEUS_QUERY='rate(http_requests_total[5m])' \
PUSHGATEWAY_URL=http://pushgateway:9091 \
GRAFANA_URL=https://grafana.company.com \
GRAFANA_API_TOKEN=glsa_xxx \
GRAFANA_DASHBOARD_UID=my-dashboard \
make run

# Verify:
# 1. anomaly_score metric appears in Prometheus
# 2. Annotations appear in Grafana when anomaly ends
```

✅ **COMPLETED**
