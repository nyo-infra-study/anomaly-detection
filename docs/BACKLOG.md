# Backlog

Unfinished work and future improvements.

## Not Yet Implemented

### Kubernetes Manifests
- Deployment, Service, ConfigMap for k8s

### CI/CD
- GitHub Actions workflow for lint/test/build
- Pre-commit hooks setup

### Dataset Download
- Anomaly detection datasets need to be downloaded from:
  - [Google Drive](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2)
  - [Hugging Face](https://huggingface.co/datasets/thuml/Time-Series-Library)
- Extract to `Time-Series-Library/dataset/`

## Ready to Use

### Training Pipeline
- `scripts/train.py` - Train TimesNet using Time-Series-Library
- `scripts/export_onnx.py` - Export to ONNX for faster inference

```bash
# 1. Train on SMD dataset (requires dataset download first)
python scripts/train.py --dataset SMD

# 2. Export to ONNX
python scripts/export_onnx.py
```

## Future Improvements

### Model
- [ ] Pattern-based models (one model per service pattern)
- [ ] Online learning / periodic retraining

### Observability
- [ ] Prometheus metrics for the service itself (inference latency, errors)
- [ ] Structured error tracking (Sentry integration)

### Scaling
- [ ] Batch inference for multiple queries
- [ ] Horizontal scaling with leader election for Grafana annotations
