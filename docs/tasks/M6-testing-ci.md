# M6: Testing & CI

## Goal
Add comprehensive tests and GitHub Actions CI pipeline.

**Depends on**: M5 (Production Hardening)

---

## Tasks

### 6.1 Set up pytest configuration

**What**: Configure pytest with async support and coverage

**Deliverable**:
```toml
# In pyproject.toml, add:
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
addopts = "-v --tb=short"

[tool.coverage.run]
source = ["src/anomaly_detection"]
branch = true

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "if __name__ == .__main__.:",
    "raise NotImplementedError",
]
```

Add coverage dependency:
```bash
uv add --dev pytest-cov
```

---

### 6.2 Write integration tests

**What**: Test full flow with mocked external services

**Deliverable**:
```python
# tests/test_integration.py
import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
import numpy as np


class TestFullPipeline:
    """Integration tests for the full anomaly detection pipeline."""

    @pytest.fixture
    def mock_prometheus_response(self):
        """Simulated Prometheus query_range response."""
        return [
            {
                "metric": {"__name__": "http_requests", "job": "api"},
                "values": [
                    [1000 + i * 60, str(100 + np.random.randn() * 10)]
                    for i in range(96)
                ],
            },
            {
                "metric": {"__name__": "http_errors", "job": "api"},
                "values": [
                    [1000 + i * 60, str(5 + np.random.randn() * 2)]
                    for i in range(96)
                ],
            },
        ]

    @pytest.mark.asyncio
    async def test_data_fetch_and_preprocess(self, mock_prometheus_response):
        """Test: Prometheus → preprocess → ready for inference."""
        from anomaly_detection.detector.preprocessor import (
            prometheus_to_array,
            normalize,
            pad_or_truncate,
        )

        # Convert response
        data, names = prometheus_to_array(mock_prometheus_response)
        
        assert data.shape == (2, 96)
        assert len(names) == 2
        
        # Normalize
        data_norm, means, stds = normalize(data)
        
        # Check normalization worked
        assert abs(data_norm.mean()) < 0.1  # Should be close to 0
        assert abs(data_norm.std() - 1.0) < 0.1  # Should be close to 1

    @pytest.mark.asyncio
    async def test_detector_inference(self):
        """Test: Mock detector produces valid scores."""
        from anomaly_detection.detector.mock import MockDetector
        
        detector = MockDetector(anomaly_probability=0.2)
        detector.load()
        
        data = np.random.randn(5, 96).astype(np.float32)
        scores = detector.predict(data)
        
        assert scores.shape == (5,)
        assert all(0 <= s <= 1 for s in scores)

    @pytest.mark.asyncio
    async def test_output_metrics(self):
        """Test: Metrics are updated without errors."""
        from anomaly_detection.output.metrics import update_scores
        
        scores = {
            "metric_a": 0.3,
            "metric_b": 0.85,
            "metric_c": 0.5,
        }
        
        # Should not raise
        update_scores(scores, threshold=0.75)

    @pytest.mark.asyncio
    async def test_grafana_state_machine(self):
        """Test: Anomaly annotator tracks state correctly."""
        from anomaly_detection.output.grafana import AnomalyAnnotator
        from datetime import datetime, timedelta
        
        annotator = AnomalyAnnotator(grafana_url=None, api_token=None)
        annotator.states = {}
        
        now = datetime.utcnow()
        
        # Normal → nothing happens
        await annotator.update("m1", 0.3, threshold=0.75, timestamp=now)
        assert "m1" not in annotator.states or not annotator.states.get("m1", MagicMock()).is_anomalous
        
        # Spike → anomaly starts
        await annotator.update("m1", 0.9, threshold=0.75, timestamp=now + timedelta(minutes=1))
        assert annotator.states["m1"].is_anomalous
        assert annotator.states["m1"].started_at is not None
        
        # Still high → still anomalous
        await annotator.update("m1", 0.85, threshold=0.75, timestamp=now + timedelta(minutes=2))
        assert annotator.states["m1"].is_anomalous
        
        # Drops → anomaly ends (would write annotation if enabled)
        await annotator.update("m1", 0.4, threshold=0.75, timestamp=now + timedelta(minutes=3))
        assert not annotator.states["m1"].is_anomalous
        
        await annotator.close()
```

---

### 6.3 Write unit tests for config

**What**: Test settings loading

**Deliverable**:
```python
# tests/test_config.py
import os
import pytest


class TestConfig:
    def test_default_values(self):
        """Config has sensible defaults."""
        from anomaly_detection.config import Settings
        
        settings = Settings()
        
        assert settings.prometheus_url == "http://localhost:9090"
        assert settings.window_size == 96
        assert settings.anomaly_threshold == 0.75

    def test_env_override(self, monkeypatch):
        """Environment variables override defaults."""
        monkeypatch.setenv("PROMETHEUS_URL", "http://custom:9090")
        monkeypatch.setenv("ANOMALY_THRESHOLD", "0.8")
        
        from anomaly_detection.config import Settings
        
        settings = Settings()
        
        assert settings.prometheus_url == "http://custom:9090"
        assert settings.anomaly_threshold == 0.8
```

---

### 6.4 Add test fixtures

**What**: Shared fixtures for tests

**Deliverable**:
```python
# tests/conftest.py
import pytest
import numpy as np
from datetime import datetime


@pytest.fixture
def sample_data():
    """Sample preprocessed data for testing."""
    np.random.seed(42)
    return np.random.randn(5, 96).astype(np.float32)


@pytest.fixture
def sample_metric_names():
    """Sample metric names."""
    return [
        "http_requests_total",
        "http_errors_total",
        "response_time_p99",
        "cpu_usage",
        "memory_usage",
    ]


@pytest.fixture
def sample_scores():
    """Sample anomaly scores."""
    return {
        "http_requests_total": 0.3,
        "http_errors_total": 0.9,
        "response_time_p99": 0.5,
        "cpu_usage": 0.2,
        "memory_usage": 0.85,
    }


@pytest.fixture
def prometheus_response():
    """Mock Prometheus API response."""
    return [
        {
            "metric": {"__name__": "test_metric", "job": "test"},
            "values": [[1000 + i * 60, str(i)] for i in range(96)],
        }
    ]
```

---

### 6.5 Create GitHub Actions workflow

**What**: CI pipeline for lint, test, build

**Deliverable**:
```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Install uv
        uses: astral-sh/setup-uv@v3
      
      - name: Set up Python
        run: uv python install 3.11
      
      - name: Install dependencies
        run: uv sync --dev
      
      - name: Lint with ruff
        run: uv run ruff check src/ tests/
      
      - name: Type check with mypy
        run: uv run mypy src/

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Install uv
        uses: astral-sh/setup-uv@v3
      
      - name: Set up Python
        run: uv python install 3.11
      
      - name: Install dependencies
        run: uv sync --dev
      
      - name: Run tests with coverage
        run: uv run pytest tests/ --cov=src/anomaly_detection --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          files: coverage.xml
        continue-on-error: true

  build:
    runs-on: ubuntu-latest
    needs: [lint, test]
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      
      - name: Build Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          push: false
          tags: anomaly-detection:test
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

---

### 6.6 Add pre-commit hooks

**What**: Run linters before commit

**Deliverable**:
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.4
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=1000']
```

Install:
```bash
uv add --dev pre-commit
uv run pre-commit install
```

---

### 6.7 Update Makefile with test targets

**What**: Add test commands to Makefile

**Deliverable**:
```makefile
# Makefile (updated)
.PHONY: install run test test-cov lint fmt clean pre-commit

install:
	uv sync

run:
	uv run python -m anomaly_detection.main

test:
	uv run pytest tests/ -v

test-cov:
	uv run pytest tests/ --cov=src/anomaly_detection --cov-report=html
	@echo "Coverage report: htmlcov/index.html"

lint:
	uv run ruff check src/ tests/
	uv run mypy src/

fmt:
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

pre-commit:
	uv run pre-commit run --all-files

clean:
	rm -rf .venv/ __pycache__/ .pytest_cache/ .mypy_cache/ .ruff_cache/ htmlcov/ .coverage
```

---

## Checklist

- [ ] 6.1 pytest configured
- [ ] 6.2 Integration tests pass
- [ ] 6.3 Config tests pass
- [ ] 6.4 Fixtures created
- [ ] 6.5 GitHub Actions workflow works
- [ ] 6.6 Pre-commit hooks installed
- [ ] 6.7 Makefile updated

## Done When

```bash
# All tests pass
make test

# Coverage report generated
make test-cov

# Lint passes
make lint

# CI passes on GitHub
git push  # → check Actions tab
```

---

## Test Coverage Goals

| Module | Target | Notes |
|--------|--------|-------|
| `config.py` | 100% | Simple, should be fully covered |
| `preprocessor.py` | 90%+ | Core logic, well tested |
| `prometheus.py` | 80%+ | HTTP mocking needed |
| `grafana.py` | 80%+ | State machine tests |
| `mock.py` | 100% | Test utility |
| `timesnet.py` | 70%+ | Needs real model for full coverage |
| `main.py` | 60%+ | Integration tests cover main flow |
