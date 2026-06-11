# M3: Detector Integration

## Goal
Load a trained TimesNet model and run inference to get anomaly scores.

**Depends on**: M2 (Data Pipeline)

---

## Tasks

### 3.1 Add PyTorch dependency

**What**: Add torch to dependencies

```bash
uv add torch --extra-index-url https://download.pytorch.org/whl/cpu
```

> **Note**: Use CPU-only torch for smaller image. Add CUDA version if you have GPU.

---

### 3.2 Implement TimesNet wrapper

**What**: Load model weights and run inference

**Deliverable**:
```python
# src/anomaly_detection/detector/timesnet.py
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from anomaly_detection.config import settings
from anomaly_detection.utils.logging import get_logger

log = get_logger("timesnet")


class TimesNetDetector:
    """
    Wrapper for TimesNet anomaly detection inference.
    
    The model reconstructs the input — high reconstruction error = anomaly.
    """

    def __init__(self, model_path: str | None = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.model_path = Path(model_path or settings.model_path)
        self.model: nn.Module | None = None
        self.threshold = settings.anomaly_threshold

    def load(self) -> None:
        """Load model weights from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        log.info("loading model", path=str(self.model_path))
        
        # Load the full model or just state dict
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        if isinstance(checkpoint, nn.Module):
            self.model = checkpoint
        else:
            # Assume it's a state dict — need to reconstruct model architecture
            # This depends on how you saved the model during training
            raise NotImplementedError(
                "State dict loading requires model architecture. "
                "Either save full model with torch.save(model, path) "
                "or implement architecture here."
            )
        
        self.model.eval()
        log.info("model loaded", device=str(self.device))

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Run inference on preprocessed data.
        
        Args:
            data: shape (num_metrics, seq_len) — normalized
        
        Returns:
            scores: shape (num_metrics,) — anomaly scores in [0, 1]
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Convert to tensor: (batch, seq_len, features)
        # TimesNet expects (B, L, C) where C = num_features
        # For univariate per-metric, we treat each metric as a separate batch item
        x = torch.from_numpy(data).float().to(self.device)
        x = x.unsqueeze(-1)  # (num_metrics, seq_len, 1)

        with torch.no_grad():
            # Forward pass — get reconstruction
            reconstruction = self.model(x)
            
            # Reconstruction error (MSE per sample)
            mse = ((x - reconstruction) ** 2).mean(dim=(1, 2))  # (num_metrics,)
            
            # Convert to score in [0, 1]
            # Using sigmoid to squash — tune the scaling factor based on your model
            scores = torch.sigmoid(mse * 10).cpu().numpy()

        return scores

    def predict_with_labels(
        self,
        data: np.ndarray,
        metric_names: list[str],
    ) -> dict[str, float]:
        """Convenience: return dict mapping metric name to score."""
        scores = self.predict(data)
        return dict(zip(metric_names, scores.tolist()))
```

---

### 3.3 Create mock detector for testing

**What**: Detector that works without a real model (for development)

**Deliverable**:
```python
# src/anomaly_detection/detector/mock.py
import numpy as np
from anomaly_detection.utils.logging import get_logger

log = get_logger("mock_detector")


class MockDetector:
    """
    Mock detector for testing without a trained model.
    Returns random scores with occasional "anomalies".
    """

    def __init__(self, anomaly_probability: float = 0.1):
        self.anomaly_probability = anomaly_probability

    def load(self) -> None:
        log.info("mock detector ready")

    def predict(self, data: np.ndarray) -> np.ndarray:
        num_metrics = data.shape[0]
        
        # Generate mostly low scores with occasional high ones
        scores = np.random.uniform(0.1, 0.4, size=num_metrics)
        
        # Randomly make some anomalous
        anomaly_mask = np.random.random(num_metrics) < self.anomaly_probability
        scores[anomaly_mask] = np.random.uniform(0.8, 0.95, size=anomaly_mask.sum())
        
        return scores.astype(np.float32)

    def predict_with_labels(
        self,
        data: np.ndarray,
        metric_names: list[str],
    ) -> dict[str, float]:
        scores = self.predict(data)
        return dict(zip(metric_names, scores.tolist()))
```

---

### 3.4 Add detector factory

**What**: Choose real or mock detector based on config

**Deliverable**:
```python
# src/anomaly_detection/detector/__init__.py
from pathlib import Path
from anomaly_detection.config import settings
from anomaly_detection.utils.logging import get_logger

log = get_logger("detector")


def get_detector():
    """
    Factory: return real TimesNet if model exists, else mock.
    """
    model_path = Path(settings.model_path)
    
    if model_path.exists():
        from anomaly_detection.detector.timesnet import TimesNetDetector
        log.info("using TimesNet detector", model_path=str(model_path))
        return TimesNetDetector()
    else:
        from anomaly_detection.detector.mock import MockDetector
        log.warning(
            "model not found, using mock detector",
            expected_path=str(model_path),
        )
        return MockDetector()
```

---

### 3.5 Integrate detector into main loop

**What**: Run inference in the main loop

**Deliverable**: Update `main.py`:
```python
# src/anomaly_detection/main.py
import asyncio
from anomaly_detection.config import settings
from anomaly_detection.utils.logging import setup_logging, get_logger
from anomaly_detection.data.prometheus import PrometheusClient
from anomaly_detection.detector import get_detector
from anomaly_detection.detector.preprocessor import (
    prometheus_to_array,
    normalize,
    pad_or_truncate,
)

log = get_logger("main")


async def run_loop():
    client = PrometheusClient()
    detector = get_detector()
    detector.load()

    try:
        while True:
            log.info("cycle start")
            
            # 1. Fetch data
            result = await client.fetch_window(
                query=settings.prometheus_query,
                window_minutes=settings.window_size,
            )
            
            # 2. Preprocess
            data, metric_names = prometheus_to_array(result)
            if data.size == 0:
                log.warning("no data returned")
                await asyncio.sleep(settings.fetch_interval_seconds)
                continue
            
            data = pad_or_truncate(data, settings.window_size)
            data_norm, means, stds = normalize(data)
            
            # 3. Run inference
            scores = detector.predict_with_labels(data_norm, metric_names)
            
            # Log results
            for metric, score in scores.items():
                severity = "high" if score > settings.anomaly_threshold else "low"
                log.info(
                    "score",
                    metric=metric,
                    score=round(score, 3),
                    severity=severity,
                )
            
            # TODO M4: push to prometheus + grafana
            
            await asyncio.sleep(settings.fetch_interval_seconds)
            
    finally:
        await client.close()


def main() -> None:
    setup_logging()
    log.info("anomaly-detection starting")
    
    try:
        asyncio.run(run_loop())
    except KeyboardInterrupt:
        log.info("shutting down")


if __name__ == "__main__":
    main()
```

---

### 3.6 Write detector tests

**What**: Unit tests for detector

**Deliverable**:
```python
# tests/test_detector.py
import numpy as np
import pytest
from anomaly_detection.detector.mock import MockDetector


class TestMockDetector:
    def test_predict_shape(self):
        detector = MockDetector()
        detector.load()
        
        data = np.random.randn(5, 96).astype(np.float32)
        scores = detector.predict(data)
        
        assert scores.shape == (5,)
        assert all(0 <= s <= 1 for s in scores)

    def test_predict_with_labels(self):
        detector = MockDetector()
        detector.load()
        
        data = np.random.randn(3, 96).astype(np.float32)
        names = ["metric_a", "metric_b", "metric_c"]
        
        result = detector.predict_with_labels(data, names)
        
        assert set(result.keys()) == set(names)
        assert all(isinstance(v, float) for v in result.values())
```

---

## Checklist

- [x] 3.1 PyTorch added (lazy import)
- [x] 3.2 `TimesNetDetector` implemented
- [x] 3.3 `MockDetector` implemented
- [x] 3.4 Detector factory works
- [x] 3.5 Main loop runs inference
- [x] 3.6 Tests passing

## Done When

```bash
make run
# Logs show "score" entries for each metric with severity
```

✅ **COMPLETED**

## Next: Training a Real Model

The `MockDetector` lets you develop the full pipeline. To use a real model:

1. Train using `scripts/train.py` (copies from Time-Series-Library)
2. Export model: `torch.save(model, "models/timesnet.pt")`
3. Restart service — it auto-detects the model file
