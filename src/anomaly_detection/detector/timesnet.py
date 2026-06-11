"""TimesNet model wrapper for inference."""

from pathlib import Path
from typing import Any

import numpy as np

from anomaly_detection.config import settings
from anomaly_detection.utils.logging import get_logger

log = get_logger("timesnet")


class TimesNetDetector:
    """
    Wrapper for TimesNet anomaly detection inference.

    The model reconstructs the input — high reconstruction error = anomaly.
    """

    def __init__(self, model_path: str | None = None, device: str = "cpu"):
        self.device = device
        self.model_path = Path(model_path or settings.model_path)
        self.model: Any = None
        self.threshold = settings.anomaly_threshold

    def load(self) -> None:
        """Load model weights from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        log.info("loading model", path=str(self.model_path))

        # Lazy import torch only when actually using TimesNet
        try:
            import torch
        except ImportError as e:
            raise ImportError(
                "PyTorch not installed. Run: uv add torch --extra-index-url "
                "https://download.pytorch.org/whl/cpu"
            ) from e

        # Load the full model or state dict
        checkpoint = torch.load(self.model_path, map_location=self.device)

        if isinstance(checkpoint, torch.nn.Module):
            self.model = checkpoint
        else:
            # State dict requires reconstructing architecture
            raise NotImplementedError(
                "State dict loading requires model architecture. "
                "Either save full model with torch.save(model, path) "
                "or implement architecture here."
            )

        self.model.eval()
        log.info("model loaded", device=self.device)

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

        import torch

        # Convert to tensor: (batch, seq_len, features)
        # TimesNet expects (B, L, C) where C = num_features
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
