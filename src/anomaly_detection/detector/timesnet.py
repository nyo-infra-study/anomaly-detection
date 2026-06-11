"""TimesNet model wrapper for inference (PyTorch and ONNX)."""

from pathlib import Path
from typing import Any

import numpy as np

from anomaly_detection.config import settings
from anomaly_detection.utils.logging import get_logger

log = get_logger("timesnet")


class TimesNetDetector:
    """
    Wrapper for TimesNet anomaly detection inference.

    Supports both PyTorch (.pt) and ONNX (.onnx) models.
    The model reconstructs the input — high reconstruction error = anomaly.
    
    ONNX is preferred for production:
    - 2-3x faster inference
    - ~50MB runtime vs ~2GB for PyTorch
    - No Python GIL issues
    """

    def __init__(self, model_path: str | None = None, device: str = "cpu"):
        self.device = device
        self.model_path = Path(model_path or settings.model_path)
        self.model: Any = None
        self.onnx_session: Any = None
        self.is_onnx = False
        self.threshold = settings.anomaly_threshold

    def load(self) -> None:
        """Load model weights from disk (PyTorch or ONNX)."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        # Detect model format
        suffix = self.model_path.suffix.lower()
        
        if suffix == ".onnx":
            self._load_onnx()
        elif suffix in (".pt", ".pth"):
            self._load_pytorch()
        else:
            raise ValueError(f"Unknown model format: {suffix}. Use .onnx or .pt")

    def _load_onnx(self) -> None:
        """Load ONNX model."""
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "onnxruntime not installed. Run: uv add onnxruntime"
            ) from e
        
        log.info("loading ONNX model", path=str(self.model_path))
        
        # Configure session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Select execution provider
        providers = ["CPUExecutionProvider"]
        if self.device != "cpu":
            # Try GPU providers
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            elif "CoreMLExecutionProvider" in available:  # pragma: no cover
                providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        
        self.onnx_session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=providers,
        )
        self.is_onnx = True
        
        # Get input/output names
        self._input_name = self.onnx_session.get_inputs()[0].name
        self._output_name = self.onnx_session.get_outputs()[0].name
        
        log.info(
            "ONNX model loaded",
            providers=self.onnx_session.get_providers(),
            input_name=self._input_name,
        )

    def _load_pytorch(self) -> None:  # pragma: no cover
        """Load PyTorch model."""
        try:
            import torch
        except ImportError as e:
            raise ImportError(
                "PyTorch not installed. Run: uv add torch --extra-index-url "
                "https://download.pytorch.org/whl/cpu"
            ) from e

        log.info("loading PyTorch model", path=str(self.model_path))

        # Load the full model or state dict
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

        if isinstance(checkpoint, torch.nn.Module):
            self.model = checkpoint
        else:
            # State dict requires reconstructing architecture
            raise NotImplementedError(
                "State dict loading requires model architecture. "
                "Either save full model with torch.save(model, path) "
                "or export to ONNX: python scripts/export_onnx.py"
            )

        self.model.eval()
        self.is_onnx = False
        log.info("PyTorch model loaded", device=self.device)

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Run inference on preprocessed data.

        Args:
            data: shape (num_metrics, seq_len) — normalized

        Returns:
            scores: shape (num_metrics,) — anomaly scores in [0, 1]
        """
        if self.is_onnx:
            return self._predict_onnx(data)
        else:
            return self._predict_pytorch(data)  # pragma: no cover

    def _predict_onnx(self, data: np.ndarray) -> np.ndarray:
        """Run ONNX inference."""
        if self.onnx_session is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Prepare input: (batch, seq_len, features)
        # data comes as (num_metrics, seq_len), need (num_metrics, seq_len, 1)
        x = data.astype(np.float32)
        if x.ndim == 2:
            x = x[:, :, np.newaxis]  # Add feature dimension
        
        # Run inference
        outputs = self.onnx_session.run(
            [self._output_name],
            {self._input_name: x},
        )
        reconstruction = outputs[0]
        
        # Compute reconstruction error (MSE per sample)
        mse = np.mean((x - reconstruction) ** 2, axis=(1, 2))
        
        # Convert to score in [0, 1] using sigmoid
        scores = 1 / (1 + np.exp(-mse * 10))
        
        return scores

    def _predict_pytorch(self, data: np.ndarray) -> np.ndarray:  # pragma: no cover
        """Run PyTorch inference."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        import torch

        # Convert to tensor: (batch, seq_len, features)
        x = torch.from_numpy(data).float().to(self.device)
        x = x.unsqueeze(-1)  # (num_metrics, seq_len, 1)

        with torch.no_grad():
            # Forward pass — get reconstruction
            reconstruction = self.model(x)

            # Reconstruction error (MSE per sample)
            mse = ((x - reconstruction) ** 2).mean(dim=(1, 2))

            # Convert to score in [0, 1]
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
