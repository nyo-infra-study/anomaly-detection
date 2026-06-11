# M5: Production Hardening

## Goal
Make the service production-ready with Docker, health checks, graceful shutdown, and proper error handling.

**Depends on**: M4 (Output Pipeline)

---

## Tasks

### 5.1 Create Dockerfile

**What**: Multi-stage build with uv

**Deliverable**:
```dockerfile
# Dockerfile
FROM python:3.11-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies (no dev deps)
RUN uv sync --frozen --no-dev

# Copy source
COPY src/ src/

# ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy virtual env from builder
COPY --from=builder /app/.venv /app/.venv

# Copy source and models
COPY --from=builder /app/src /app/src
COPY models/ models/

# Set path to use venv
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8080/health').raise_for_status()"

# Run
CMD ["python", "-m", "anomaly_detection.main"]
```

---

### 5.2 Add health endpoint

**What**: Simple HTTP server for k8s probes

**Deliverable**:
```python
# src/anomaly_detection/health.py
import asyncio
from aiohttp import web
from anomaly_detection.utils.logging import get_logger

log = get_logger("health")

# Global state for health checks
_healthy = True
_last_successful_cycle: float = 0


def set_healthy(healthy: bool) -> None:
    global _healthy
    _healthy = healthy


def record_successful_cycle() -> None:
    global _last_successful_cycle
    _last_successful_cycle = asyncio.get_event_loop().time()


async def health_handler(request: web.Request) -> web.Response:
    if _healthy:
        return web.json_response({"status": "healthy"})
    return web.json_response({"status": "unhealthy"}, status=503)


async def ready_handler(request: web.Request) -> web.Response:
    """Ready = has completed at least one cycle recently."""
    current_time = asyncio.get_event_loop().time()
    if _last_successful_cycle > 0 and (current_time - _last_successful_cycle) < 300:
        return web.json_response({"status": "ready"})
    return web.json_response({"status": "not ready"}, status=503)


async def start_health_server(port: int = 8080) -> web.AppRunner:
    """Start health check HTTP server."""
    app = web.Application()
    app.router.add_get("/health", health_handler)
    app.router.add_get("/ready", ready_handler)
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    
    log.info("health server started", port=port)
    return runner
```

Add `aiohttp` dependency:
```bash
uv add aiohttp
```

---

### 5.3 Add graceful shutdown

**What**: Handle SIGTERM/SIGINT properly

**Deliverable**: Update `main.py`:
```python
# src/anomaly_detection/main.py
import asyncio
import signal
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
from anomaly_detection.health import (
    start_health_server,
    set_healthy,
    record_successful_cycle,
)

log = get_logger("main")

# Shutdown flag
_shutdown_event: asyncio.Event | None = None


def handle_shutdown(sig: signal.Signals) -> None:
    log.info("received shutdown signal", signal=sig.name)
    if _shutdown_event:
        _shutdown_event.set()


async def run_loop():
    global _shutdown_event
    _shutdown_event = asyncio.Event()
    
    # Register signal handlers
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda s=sig: handle_shutdown(s))
    
    # Start health server
    health_runner = await start_health_server(port=8080)
    
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
                    await asyncio.wait_for(
                        _shutdown_event.wait(),
                        timeout=settings.fetch_interval_seconds,
                    )
                    continue
                
                data = pad_or_truncate(data, settings.window_size)
                data_norm, means, stds = normalize(data)
                
                # 3. Inference
                scores = detector.predict_with_labels(data_norm, metric_names)
                
                # 4. Output
                update_scores(scores)
                push_metrics()
                
                for metric_name, score in scores.items():
                    await annotator.update(metric_name, score)
                
                # Mark cycle complete
                record_successful_cycle()
                
                high_count = sum(1 for s in scores.values() if s > settings.anomaly_threshold)
                log.info("cycle complete", total=len(scores), anomalies=high_count)
                
            except Exception as e:
                log.error("cycle failed", error=str(e))
                set_healthy(False)
            
            # Wait for next cycle or shutdown
            try:
                await asyncio.wait_for(
                    _shutdown_event.wait(),
                    timeout=settings.fetch_interval_seconds,
                )
            except asyncio.TimeoutError:
                pass  # Normal — continue to next cycle
                
    finally:
        log.info("cleaning up")
        set_healthy(False)
        await prom_client.close()
        await annotator.close()
        await health_runner.cleanup()


def main() -> None:
    setup_logging()
    log.info("anomaly-detection starting", version="0.1.0")
    
    asyncio.run(run_loop())
    log.info("anomaly-detection stopped")


if __name__ == "__main__":
    main()
```

---

### 5.4 Create docker-compose.yml

**What**: Local development with dependencies

**Deliverable**:
```yaml
# docker-compose.yml
version: "3.8"

services:
  anomaly-detection:
    build: .
    environment:
      - PROMETHEUS_URL=http://prometheus:9090
      - PROMETHEUS_QUERY=up
      - FETCH_INTERVAL_SECONDS=60
      - LOG_LEVEL=INFO
      - LOG_FORMAT=json
      # Optional: set these if you have Grafana
      # - GRAFANA_URL=http://grafana:3000
      # - GRAFANA_API_TOKEN=xxx
      # - GRAFANA_DASHBOARD_UID=xxx
    ports:
      - "8080:8080"
    depends_on:
      - prometheus
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
    ports:
      - "9090:9090"

  # Optional: Pushgateway for push-based metrics
  pushgateway:
    image: prom/pushgateway:latest
    ports:
      - "9091:9091"
```

---

### 5.5 Add retry logic

**What**: Retry transient failures

**Deliverable**:
```python
# src/anomaly_detection/utils/retry.py
import asyncio
from functools import wraps
from typing import Callable, TypeVar
from anomaly_detection.utils.logging import get_logger

log = get_logger("retry")

T = TypeVar("T")


def retry_async(
    max_attempts: int = 3,
    delay_seconds: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """
    Decorator for retrying async functions with exponential backoff.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None
            delay = delay_seconds
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        log.warning(
                            "retrying",
                            func=func.__name__,
                            attempt=attempt,
                            max_attempts=max_attempts,
                            delay=delay,
                            error=str(e),
                        )
                        await asyncio.sleep(delay)
                        delay *= backoff_factor
                    else:
                        log.error(
                            "all retries failed",
                            func=func.__name__,
                            error=str(e),
                        )
            
            raise last_exception
        
        return wrapper
    return decorator
```

Update Prometheus client to use retry:
```python
# In data/prometheus.py
from anomaly_detection.utils.retry import retry_async

class PrometheusClient:
    # ...
    
    @retry_async(max_attempts=3, delay_seconds=2.0)
    async def query_range(self, ...):
        # existing code
```

---

### 5.6 Add Kubernetes manifests

**What**: Deployment, Service, ConfigMap

**Deliverable**:
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: anomaly-detection
  labels:
    app: anomaly-detection
spec:
  replicas: 1
  selector:
    matchLabels:
      app: anomaly-detection
  template:
    metadata:
      labels:
        app: anomaly-detection
    spec:
      containers:
        - name: anomaly-detection
          image: your-registry/anomaly-detection:latest
          ports:
            - containerPort: 8080
              name: health
          envFrom:
            - configMapRef:
                name: anomaly-detection-config
            - secretRef:
                name: anomaly-detection-secrets
          livenessProbe:
            httpGet:
              path: /health
              port: health
            initialDelaySeconds: 10
            periodSeconds: 30
          readinessProbe:
            httpGet:
              path: /ready
              port: health
            initialDelaySeconds: 30
            periodSeconds: 10
          resources:
            requests:
              memory: "256Mi"
              cpu: "100m"
            limits:
              memory: "1Gi"
              cpu: "1000m"
          volumeMounts:
            - name: model
              mountPath: /app/models
              readOnly: true
      volumes:
        - name: model
          configMap:
            name: anomaly-detection-model
            # Or use PVC/S3 for larger models
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: anomaly-detection-config
data:
  PROMETHEUS_URL: "http://prometheus.monitoring:9090"
  PROMETHEUS_QUERY: "rate(http_requests_total[5m])"
  FETCH_INTERVAL_SECONDS: "60"
  LOG_LEVEL: "INFO"
  LOG_FORMAT: "json"
---
apiVersion: v1
kind: Secret
metadata:
  name: anomaly-detection-secrets
type: Opaque
stringData:
  GRAFANA_API_TOKEN: "your-token-here"
```

---

## Checklist

- [x] 5.1 Dockerfile builds
- [x] 5.2 Health endpoint works
- [x] 5.3 Graceful shutdown works
- [x] 5.4 docker-compose runs locally
- [x] 5.5 Retry logic added
- [ ] 5.6 K8s manifests created

## Done When

```bash
# Build and run
docker-compose up --build

# Test health
curl http://localhost:8080/health
# {"status": "healthy"}

# Test graceful shutdown
docker-compose stop
# Logs show "received shutdown signal" and "cleaning up"
```

✅ **MOSTLY COMPLETED** (K8s manifests not yet created)
