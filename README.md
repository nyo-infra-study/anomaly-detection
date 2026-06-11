# Anomaly Detection Service

TimesNet-based anomaly detection for Prometheus metrics with Datadog-style Grafana visualization.

## Quick Start

```bash
# Install dependencies
make install

# Run with mock detector (no trained model needed)
make run

# Run tests
make test
```

## Architecture

```mermaid
flowchart LR
    subgraph AD["AD Service (loop every 60s)"]
        direction LR
        Fetch["Fetch
        PromQL"] --> Preproc["Preprocess
        normalize"]
        Preproc --> TimesNet["TimesNet
        inference"]
        TimesNet --> Output["Output"]
    end

    Prom1[(Prometheus)] --> Fetch
    Output --> Prom2[(anomaly_score)]
    Output --> Grafana[Grafana API]
```

## Configuration

All config via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `PROMETHEUS_URL` | `http://localhost:9090` | Prometheus/Gigapipe URL |
| `PROMETHEUS_QUERY` | `up` | PromQL query for metrics |
| `FETCH_INTERVAL_SECONDS` | `60` | Seconds between cycles |
| `DETECTOR_TYPE` | `auto` | `auto`, `timesnet`, `statistical`, `rolling`, `mock` |
| `MODEL_PATH` | `models/timesnet.pt` | Path to trained model (for timesnet) |
| `ANOMALY_THRESHOLD` | `0.75` | Score above this = anomaly |
| `Z_THRESHOLD` | `2.5` | Z-score threshold (for statistical) |
| `EMA_ALPHA` | `0.1` | EMA smoothing (for rolling) |
| `GRAFANA_URL` | - | Grafana URL for annotations |
| `GRAFANA_API_TOKEN` | - | Grafana service account token |
| `LOG_LEVEL` | `INFO` | Logging level |
| `LOG_FORMAT` | `json` | `json` or `console` |

## Detector Options

### 1. Statistical (no training needed)
```bash
DETECTOR_TYPE=statistical make run
```
Uses Z-score: if last value is > 2.5 standard deviations from window mean → anomaly.

### 2. Rolling Statistical (maintains history)
```bash
DETECTOR_TYPE=rolling make run
```
Uses exponential moving average (EMA) for more stable baselines across cycles.

### 3. TimesNet (CNN-based, needs trained model)
```bash
# First, get a model (see scripts/download_model.py)
python scripts/download_model.py --train --dataset SMD

# Then run
DETECTOR_TYPE=timesnet make run
```

### 4. Auto (default)
```bash
make run
```
Uses TimesNet if `models/timesnet.pt` exists, otherwise falls back to statistical.

## Development

```bash
# Format code
make fmt

# Lint
make lint

# Test with coverage
make test-cov
```

## Docker

```bash
# Build and run with local Prometheus
docker-compose up --build

# Health check
curl http://localhost:8080/health
```

## Documentation

See [docs/](./docs/) for:
- [Project Plan](./docs/PLAN.md)
- [Task Breakdowns](./docs/tasks/)
