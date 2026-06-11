# M1: Project Bootstrap

## Goal
Set up the project skeleton with uv, config, and logging. No ML code yet — just the foundation.

---

## Tasks

### 1.1 Initialize uv project

**What**: Create `pyproject.toml` and initialize uv

```bash
cd anomaly-detection
uv init
```

**Deliverable**: `pyproject.toml` with:
```toml
[project]
name = "anomaly-detection"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = []

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

**Verify**:
```bash
uv sync
# Should create .venv/ and uv.lock
```

---

### 1.2 Add core dependencies

**What**: Add runtime dependencies

```bash
uv add pydantic-settings structlog httpx prometheus-client
```

**Deliverable**: `pyproject.toml` updated with dependencies

---

### 1.3 Add dev dependencies

**What**: Add testing and linting tools

```bash
uv add --dev pytest pytest-asyncio ruff mypy
```

**Deliverable**: `pyproject.toml` has `[project.optional-dependencies]` section

---

### 1.4 Create directory structure

**What**: Create the src layout

```bash
mkdir -p src/anomaly_detection/{detector,data,output,utils}
mkdir -p tests
mkdir -p models
mkdir -p scripts
touch src/anomaly_detection/__init__.py
touch src/anomaly_detection/{main,config}.py
touch src/anomaly_detection/detector/__init__.py
touch src/anomaly_detection/data/__init__.py
touch src/anomaly_detection/output/__init__.py
touch src/anomaly_detection/utils/__init__.py
```

**Deliverable**: Directory tree matches PLAN.md

---

### 1.5 Create `.python-version`

**What**: Pin Python version for uv

```bash
echo "3.11" > .python-version
```

---

### 1.6 Implement `config.py`

**What**: Type-safe settings from environment variables

**Deliverable**:
```python
# src/anomaly_detection/config.py
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """All config comes from environment variables."""

    # Data source
    prometheus_url: str = "http://localhost:9090"
    prometheus_query: str = "up"
    fetch_interval_seconds: int = 60

    # Model
    model_path: str = "models/timesnet.pt"
    window_size: int = 96
    anomaly_threshold: float = 0.75

    # Output
    pushgateway_url: str | None = None
    grafana_url: str | None = None
    grafana_api_token: str | None = None
    grafana_dashboard_uid: str | None = None

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"  # or "console"

    class Config:
        env_prefix = ""  # no prefix, e.g., PROMETHEUS_URL


settings = Settings()
```

**Verify**:
```bash
uv run python -c "from anomaly_detection.config import settings; print(settings)"
```

---

### 1.7 Implement `utils/logging.py`

**What**: Structured logging with structlog

**Deliverable**:
```python
# src/anomaly_detection/utils/logging.py
import structlog
from anomaly_detection.config import settings


def setup_logging() -> None:
    """Configure structlog for JSON or console output."""
    processors = [
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if settings.log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(structlog, settings.log_level.upper(), structlog.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a named logger."""
    return structlog.get_logger(name)
```

---

### 1.8 Create `main.py` skeleton

**What**: Entrypoint that just logs and exits

**Deliverable**:
```python
# src/anomaly_detection/main.py
from anomaly_detection.config import settings
from anomaly_detection.utils.logging import setup_logging, get_logger


def main() -> None:
    setup_logging()
    log = get_logger("main")

    log.info(
        "anomaly-detection starting",
        prometheus_url=settings.prometheus_url,
        model_path=settings.model_path,
    )

    # TODO: main loop will go here
    log.info("anomaly-detection stopped")


if __name__ == "__main__":
    main()
```

**Verify**:
```bash
uv run python -m anomaly_detection.main
# Should print JSON log lines
```

---

### 1.9 Create Makefile

**What**: Developer convenience commands

**Deliverable**:
```makefile
.PHONY: install run test lint fmt clean

install:
	uv sync

run:
	uv run python -m anomaly_detection.main

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check src/ tests/
	uv run mypy src/

fmt:
	uv run ruff format src/ tests/

clean:
	rm -rf .venv/ __pycache__/ .pytest_cache/ .mypy_cache/
```

---

### 1.10 Create `.gitignore`

**What**: Ignore generated files

**Deliverable**:
```gitignore
# Python
__pycache__/
*.py[cod]
.venv/
.mypy_cache/
.pytest_cache/
.ruff_cache/

# uv
# uv.lock is committed (reproducible builds)

# Models (large files)
models/*.pt
models/*.onnx

# IDE
.idea/
.vscode/
*.swp

# Env
.env
.env.local
```

---

## Checklist

- [ ] 1.1 `uv init` done
- [ ] 1.2 Core deps added
- [ ] 1.3 Dev deps added
- [ ] 1.4 Directory structure created
- [ ] 1.5 `.python-version` created
- [ ] 1.6 `config.py` works
- [ ] 1.7 `logging.py` works
- [ ] 1.8 `main.py` runs and logs
- [ ] 1.9 `Makefile` works (`make run`)
- [ ] 1.10 `.gitignore` created

## Done When

```bash
make run
# Outputs JSON log with "anomaly-detection starting"
```
