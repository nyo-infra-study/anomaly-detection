"""Shared test fixtures."""

import numpy as np
import pytest


@pytest.fixture
def sample_data() -> np.ndarray:
    """Sample preprocessed data for testing."""
    np.random.seed(42)
    return np.random.randn(5, 96).astype(np.float32)


@pytest.fixture
def sample_metric_names() -> list[str]:
    """Sample metric names."""
    return [
        "http_requests_total",
        "http_errors_total",
        "response_time_p99",
        "cpu_usage",
        "memory_usage",
    ]


@pytest.fixture
def sample_scores() -> dict[str, float]:
    """Sample anomaly scores."""
    return {
        "http_requests_total": 0.3,
        "http_errors_total": 0.9,
        "response_time_p99": 0.5,
        "cpu_usage": 0.2,
        "memory_usage": 0.85,
    }


@pytest.fixture
def prometheus_response() -> list[dict]:
    """Mock Prometheus API response."""
    return [
        {
            "metric": {"__name__": "test_metric", "job": "test"},
            "values": [[1000 + i * 60, str(float(i))] for i in range(96)],
        }
    ]
