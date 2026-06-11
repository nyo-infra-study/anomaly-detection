# Anomaly Detection Service — Documentation

## Overview

This service detects anomalies in Prometheus metrics using a hybrid approach:
- **TimesNet** (deep learning) for critical services
- **Statistical methods** (Z-score) for the long tail of services

## Architecture

```mermaid
graph TB
    subgraph src/anomaly_detection
        main[main.py]
        config[config.py]
        
        subgraph data
            prometheus[prometheus.py]
        end
        
        subgraph detector
            hybrid[hybrid.py]
            timesnet[timesnet.py]
            statistical[statistical.py]
        end
        
        subgraph output
            metrics[metrics.py]
            grafana[grafana.py]
        end
        
        subgraph utils
            logging[logging.py]
            retry[retry.py]
        end
    end
    
    main --> config
    main --> prometheus
    main --> hybrid
    main --> metrics
    main --> grafana
    main --> logging
    
    hybrid --> timesnet
    hybrid --> statistical
    prometheus --> retry
    grafana --> retry
```

## Main Loop

```mermaid
flowchart LR
    subgraph AD["AD Service (loop every 60s)"]
        direction LR
        Fetch["1. Fetch
        PromQL"] --> Preproc["2. Preprocess
        normalize, reshape"]
        Preproc --> Detect["3. Detect
        hybrid routing"]
        Detect --> Output["4. Output"]
    end

    Prom1[(Prometheus)] --> Fetch
    Output --> Prom2[(anomaly_score)]
    Output --> Grafana[Grafana API]
```

## Hybrid Detection Flow

```mermaid
flowchart TD
    Input[Metric Data] --> Check{Service in
    CRITICAL_SERVICES?}
    Check -->|Yes| TimesNet[TimesNet
    Deep Learning]
    Check -->|No| Statistical[Statistical
    Z-Score]
    TimesNet --> Score[Anomaly Score]
    Statistical --> Score
```

## Documents

| Document | Description |
|----------|-------------|
| [BACKLOG.md](./BACKLOG.md) | Unfinished work and future improvements |

## Tech Stack

| Component | Choice | Why |
|-----------|--------|-----|
| Package manager | `uv` | Fast, lockfile, replaces pip+venv |
| Python | 3.11+ | Match production, type hints |
| ML runtime | PyTorch | TimesNet weights |
| HTTP client | `httpx` | Async, modern |
| Config | `pydantic-settings` | Type-safe env vars |
| Logging | `structlog` | JSON logs for k8s |
| Metrics | `prometheus-client` | Push gateway compatible |
| Testing | `pytest` | Standard, 100% coverage |
| Linting | `ruff` + `mypy` | Fast, strict |

## Project Structure

```
anomaly-detection/
├── pyproject.toml              # dependencies + metadata
├── uv.lock                     # locked deps
├── Dockerfile
├── docker-compose.yml
├── Makefile
│
├── src/anomaly_detection/
│   ├── main.py                 # entrypoint
│   ├── config.py               # Settings class
│   ├── detector/               # anomaly detection
│   │   ├── hybrid.py           # routes to timesnet/statistical
│   │   ├── timesnet.py         # deep learning model
│   │   ├── statistical.py      # z-score based
│   │   └── rolling.py          # ema-based
│   ├── data/prometheus.py      # PromQL client
│   ├── output/
│   │   ├── metrics.py          # anomaly_score gauge
│   │   └── grafana.py          # annotations
│   └── utils/
│       ├── logging.py
│       └── retry.py
│
├── tests/                      # 100% coverage
├── models/                     # .pt files
├── scripts/download_model.py
└── docs/
```
