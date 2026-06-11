# Backlog

Unfinished work and future improvements.

## Not Yet Implemented

### Kubernetes Manifests
- Deployment, Service, ConfigMap for k8s

### CI/CD
- GitHub Actions workflow for lint/test/build
- Pre-commit hooks setup

### Training Pipeline
- `scripts/train.py` using Time-Series-Library
- `scripts/export_onnx.py` for optimized inference
- Currently only `scripts/download_model.py` exists

## Future Improvements

### Model
- [ ] ONNX export for faster inference without PyTorch
- [ ] Pattern-based models (one model per service pattern)
- [ ] Online learning / periodic retraining

### Observability
- [ ] Prometheus metrics for the service itself (inference latency, errors)
- [ ] Structured error tracking (Sentry integration)

### Scaling
- [ ] Batch inference for multiple queries
- [ ] Horizontal scaling with leader election for Grafana annotations
