# Anomaly Detection Service — Documentation

## Quick Links

| Document | Description |
|----------|-------------|
| [PLAN.md](./PLAN.md) | Project overview, architecture, tech stack |
| [tasks/](./tasks/) | Milestone task breakdowns |

## Milestones

| # | Milestone | Status | Description |
|---|-----------|--------|-------------|
| M1 | [Bootstrap](./tasks/M1-bootstrap.md) | ✅ | uv, config, logging |
| M2 | [Data Pipeline](./tasks/M2-data-pipeline.md) | ✅ | Prometheus client, preprocessing |
| M3 | [Detector](./tasks/M3-detector.md) | ✅ | TimesNet wrapper, inference |
| M4 | [Output](./tasks/M4-output.md) | ✅ | Prometheus metrics, Grafana annotations |
| M5 | [Production](./tasks/M5-production.md) | 🔶 | Docker, health checks, k8s (k8s pending) |
| M6 | [Testing & CI](./tasks/M6-testing-ci.md) | 🔶 | pytest, GitHub Actions (CI pending) |

## Getting Started

```bash
# 1. Clone
git clone git@github.com:nyo-infra-study/anomaly-detection.git
cd anomaly-detection

# 2. Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Start with M1
# Follow docs/tasks/M1-bootstrap.md
```
