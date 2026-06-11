# M2: Data Pipeline

## Goal
Fetch metrics from Prometheus/Gigapipe and format them for TimesNet input.

**Depends on**: M1 (Bootstrap)

---

## Tasks

### 2.1 Implement PromQL client

**What**: Async client to query Prometheus range data

**Deliverable**:
```python
# src/anomaly_detection/data/prometheus.py
import httpx
from datetime import datetime, timedelta
from anomaly_detection.config import settings
from anomaly_detection.utils.logging import get_logger

log = get_logger("prometheus")


class PrometheusClient:
    """Fetch time series data from Prometheus/VictoriaMetrics/Gigapipe."""

    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or settings.prometheus_url).rstrip("/")
        self.client = httpx.AsyncClient(timeout=30.0)

    async def query_range(
        self,
        query: str,
        start: datetime,
        end: datetime,
        step: str = "1m",
    ) -> list[dict]:
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

        return data["data"]["result"]

    async def fetch_window(
        self,
        query: str,
        window_minutes: int = 96,
        step: str = "1m",
    ) -> list[dict]:
        """Convenience: fetch the last N minutes of data."""
        end = datetime.utcnow()
        start = end - timedelta(minutes=window_minutes)
        return await self.query_range(query, start, end, step)

    async def close(self):
        await self.client.aclose()
```

**Verify**:
```python
# Quick test (requires running Prometheus)
import asyncio
from anomaly_detection.data.prometheus import PrometheusClient

async def test():
    client = PrometheusClient("http://localhost:9090")
    result = await client.fetch_window("up", window_minutes=5)
    print(result)
    await client.close()

asyncio.run(test())
```

---

### 2.2 Implement preprocessor

**What**: Convert Prometheus response to numpy arrays for TimesNet

**Deliverable**:
```python
# src/anomaly_detection/detector/preprocessor.py
import numpy as np
from anomaly_detection.utils.logging import get_logger

log = get_logger("preprocessor")


def prometheus_to_array(result: list[dict]) -> tuple[np.ndarray, list[str]]:
    """
    Convert Prometheus query_range result to numpy array.
    
    Args:
        result: List of {"metric": {...}, "values": [[ts, val], ...]}
    
    Returns:
        (data, metric_names)
        - data: shape (num_metrics, num_timestamps)
        - metric_names: list of metric identifiers
    """
    if not result:
        return np.array([]), []

    # Extract metric names (for labeling output)
    metric_names = []
    for series in result:
        labels = series["metric"]
        # Create readable name from labels
        name = labels.get("__name__", "unknown")
        extra = ",".join(f'{k}="{v}"' for k, v in labels.items() if k != "__name__")
        metric_names.append(f"{name}{{{extra}}}" if extra else name)

    # Convert to numpy
    # Assume all series have same timestamps (Prometheus guarantees this for range queries)
    arrays = []
    for series in result:
        values = [float(v[1]) for v in series["values"]]
        arrays.append(values)

    data = np.array(arrays, dtype=np.float32)
    log.debug("preprocessed", shape=data.shape, metrics=len(metric_names))

    return data, metric_names


def normalize(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Instance normalization (per-metric, per-window).
    
    Returns:
        (normalized, means, stds)
    """
    means = data.mean(axis=1, keepdims=True)
    stds = data.std(axis=1, keepdims=True)
    stds = np.where(stds == 0, 1.0, stds)  # Avoid division by zero

    normalized = (data - means) / stds
    return normalized, means.squeeze(), stds.squeeze()


def pad_or_truncate(data: np.ndarray, target_length: int) -> np.ndarray:
    """Ensure data has exactly target_length timestamps."""
    current_length = data.shape[1]

    if current_length == target_length:
        return data
    elif current_length > target_length:
        # Truncate (keep most recent)
        return data[:, -target_length:]
    else:
        # Pad with zeros at the start
        padding = np.zeros((data.shape[0], target_length - current_length), dtype=data.dtype)
        return np.concatenate([padding, data], axis=1)
```

---

### 2.3 Write unit tests for data pipeline

**What**: Test PromQL parsing and preprocessing

**Deliverable**:
```python
# tests/test_data_pipeline.py
import numpy as np
import pytest
from anomaly_detection.detector.preprocessor import (
    prometheus_to_array,
    normalize,
    pad_or_truncate,
)


class TestPrometheusToArray:
    def test_empty_result(self):
        data, names = prometheus_to_array([])
        assert data.shape == (0,)
        assert names == []

    def test_single_series(self):
        result = [
            {
                "metric": {"__name__": "http_requests", "job": "api"},
                "values": [[1000, "1.0"], [1060, "2.0"], [1120, "3.0"]],
            }
        ]
        data, names = prometheus_to_array(result)
        assert data.shape == (1, 3)
        assert names == ['http_requests{job="api"}']
        np.testing.assert_array_equal(data[0], [1.0, 2.0, 3.0])

    def test_multiple_series(self):
        result = [
            {"metric": {"__name__": "m1"}, "values": [[1, "1"], [2, "2"]]},
            {"metric": {"__name__": "m2"}, "values": [[1, "3"], [2, "4"]]},
        ]
        data, names = prometheus_to_array(result)
        assert data.shape == (2, 2)
        assert names == ["m1", "m2"]


class TestNormalize:
    def test_basic_normalization(self):
        data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        norm, means, stds = normalize(data)
        
        # Mean should be ~0, std should be ~1
        np.testing.assert_almost_equal(norm.mean(), 0, decimal=5)
        np.testing.assert_almost_equal(norm.std(), 1, decimal=5)

    def test_constant_series(self):
        # Edge case: all values the same
        data = np.array([[5.0, 5.0, 5.0, 5.0]])
        norm, means, stds = normalize(data)
        
        # Should not produce NaN
        assert not np.isnan(norm).any()
        np.testing.assert_array_equal(norm, [[0, 0, 0, 0]])


class TestPadOrTruncate:
    def test_exact_length(self):
        data = np.array([[1, 2, 3]])
        result = pad_or_truncate(data, 3)
        np.testing.assert_array_equal(result, data)

    def test_truncate(self):
        data = np.array([[1, 2, 3, 4, 5]])
        result = pad_or_truncate(data, 3)
        np.testing.assert_array_equal(result, [[3, 4, 5]])  # Keep most recent

    def test_pad(self):
        data = np.array([[1, 2, 3]])
        result = pad_or_truncate(data, 5)
        np.testing.assert_array_equal(result, [[0, 0, 1, 2, 3]])  # Pad at start
```

**Verify**:
```bash
make test
```

---

### 2.4 Add data fetch to main loop

**What**: Wire up the data fetching in main.py

**Deliverable**: Update `main.py`:
```python
# src/anomaly_detection/main.py
import asyncio
from anomaly_detection.config import settings
from anomaly_detection.utils.logging import setup_logging, get_logger
from anomaly_detection.data.prometheus import PrometheusClient
from anomaly_detection.detector.preprocessor import (
    prometheus_to_array,
    normalize,
    pad_or_truncate,
)

log = get_logger("main")


async def run_loop():
    client = PrometheusClient()

    try:
        while True:
            log.info("fetching metrics")
            
            # Fetch data
            result = await client.fetch_window(
                query=settings.prometheus_query,
                window_minutes=settings.window_size,
            )
            
            # Preprocess
            data, metric_names = prometheus_to_array(result)
            if data.size == 0:
                log.warning("no data returned from prometheus")
                await asyncio.sleep(settings.fetch_interval_seconds)
                continue
            
            data = pad_or_truncate(data, settings.window_size)
            data_norm, means, stds = normalize(data)
            
            log.info(
                "data ready",
                shape=data_norm.shape,
                metrics=len(metric_names),
            )
            
            # TODO M3: run detector
            # TODO M4: push output
            
            await asyncio.sleep(settings.fetch_interval_seconds)
            
    finally:
        await client.close()


def main() -> None:
    setup_logging()
    log.info(
        "anomaly-detection starting",
        prometheus_url=settings.prometheus_url,
        query=settings.prometheus_query,
    )
    
    try:
        asyncio.run(run_loop())
    except KeyboardInterrupt:
        log.info("shutting down")


if __name__ == "__main__":
    main()
```

---

## Checklist

- [x] 2.1 `PrometheusClient` implemented
- [x] 2.2 `preprocessor.py` implemented
- [x] 2.3 Unit tests passing
- [x] 2.4 Main loop fetches and logs data shape

## Done When

```bash
PROMETHEUS_URL=http://your-prometheus:9090 \
PROMETHEUS_QUERY='rate(http_requests_total[5m])' \
make run

# Logs show "data ready" with correct shape
```

✅ **COMPLETED**
