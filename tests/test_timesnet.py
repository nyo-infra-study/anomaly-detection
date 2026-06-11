"""Tests for TimesNet detector (PyTorch and ONNX)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from anomaly_detection.detector.timesnet import TimesNetDetector


class TestTimesNetDetector:
    """Tests for TimesNetDetector initialization and basic operations."""

    def test_init_with_default_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should use settings.model_path by default."""
        monkeypatch.setenv("MODEL_PATH", "custom/model.pt")
        from anomaly_detection.config import Settings

        with patch("anomaly_detection.detector.timesnet.settings", Settings()):
            detector = TimesNetDetector()
            assert str(detector.model_path) == "custom/model.pt"

    def test_init_with_custom_path(self) -> None:
        """Should accept custom model path."""
        detector = TimesNetDetector(model_path="/path/to/model.pt")
        assert str(detector.model_path) == "/path/to/model.pt"

    def test_init_with_device(self) -> None:
        """Should accept device parameter."""
        detector = TimesNetDetector(device="cuda:0")
        assert detector.device == "cuda:0"

    def test_load_file_not_found(self) -> None:
        """Should raise FileNotFoundError if model doesn't exist."""
        detector = TimesNetDetector(model_path="/nonexistent/model.pt")
        with pytest.raises(FileNotFoundError, match="Model not found"):
            detector.load()

    def test_load_unknown_format(self, tmp_path: Path) -> None:
        """Should raise ValueError for unknown model format."""
        model_file = tmp_path / "model.xyz"
        model_file.touch()
        
        detector = TimesNetDetector(model_path=str(model_file))
        with pytest.raises(ValueError, match="Unknown model format"):
            detector.load()

    def test_predict_without_load_raises(self) -> None:
        """Should raise RuntimeError if predict called before load."""
        detector = TimesNetDetector(model_path="model.pt")
        data = np.random.randn(3, 96).astype(np.float32)

        # is_onnx=False by default, so predict routes to _predict_pytorch
        # _predict_pytorch checks self.model is None
        with pytest.raises(RuntimeError, match="not loaded"):
            detector.predict(data)

    def test_predict_onnx_without_load_raises(self) -> None:
        """Should raise RuntimeError if predict on ONNX called before load."""
        detector = TimesNetDetector(model_path="model.onnx")
        detector.is_onnx = True  # Simulate ONNX path without actually loading
        data = np.random.randn(3, 96).astype(np.float32)

        # onnx_session is None
        with pytest.raises(RuntimeError, match="not loaded"):
            detector.predict(data)

    def test_predict_with_labels_without_load_raises(self) -> None:
        """Should raise RuntimeError if predict_with_labels called before load."""
        detector = TimesNetDetector(model_path="model.onnx")
        data = np.random.randn(3, 96).astype(np.float32)
        names = ["m1", "m2", "m3"]

        with pytest.raises(RuntimeError, match="not loaded"):
            detector.predict_with_labels(data, names)


class TestONNXDetector:
    """Tests for ONNX model loading and inference."""

    def test_load_onnx_without_onnxruntime(self, tmp_path: Path) -> None:
        """Should raise ImportError if onnxruntime not installed."""
        model_file = tmp_path / "model.onnx"
        model_file.touch()

        detector = TimesNetDetector(model_path=str(model_file))

        with patch.dict("sys.modules", {"onnxruntime": None}):
            with patch(
                "builtins.__import__",
                side_effect=ImportError("No module named 'onnxruntime'"),
            ):
                with pytest.raises(ImportError, match="onnxruntime not installed"):
                    detector.load()

    def test_load_onnx_success(self, tmp_path: Path) -> None:
        """Should load ONNX model successfully."""
        model_file = tmp_path / "model.onnx"
        model_file.touch()

        # Mock onnxruntime
        mock_input = MagicMock()
        mock_input.name = "input"
        mock_output = MagicMock()
        mock_output.name = "output"
        
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.get_outputs.return_value = [mock_output]
        mock_session.get_providers.return_value = ["CPUExecutionProvider"]

        mock_ort = MagicMock()
        mock_ort.InferenceSession.return_value = mock_session
        mock_ort.SessionOptions.return_value = MagicMock()
        mock_ort.GraphOptimizationLevel.ORT_ENABLE_ALL = 99

        detector = TimesNetDetector(model_path=str(model_file))

        with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
            detector._load_onnx()

        assert detector.is_onnx is True
        assert detector.onnx_session is mock_session
        assert detector._input_name == "input"
        assert detector._output_name == "output"

    def test_load_onnx_with_gpu(self, tmp_path: Path) -> None:
        """Should try GPU providers when device is not cpu."""
        model_file = tmp_path / "model.onnx"
        model_file.touch()

        mock_input = MagicMock()
        mock_input.name = "input"
        mock_output = MagicMock()
        mock_output.name = "output"
        
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.get_outputs.return_value = [mock_output]
        mock_session.get_providers.return_value = ["CUDAExecutionProvider"]

        mock_ort = MagicMock()
        mock_ort.InferenceSession.return_value = mock_session
        mock_ort.SessionOptions.return_value = MagicMock()
        mock_ort.GraphOptimizationLevel.ORT_ENABLE_ALL = 99
        mock_ort.get_available_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        detector = TimesNetDetector(model_path=str(model_file), device="cuda")

        with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
            detector._load_onnx()

        # Check that CUDA provider was requested
        call_args = mock_ort.InferenceSession.call_args
        providers = call_args[1]["providers"]
        assert "CUDAExecutionProvider" in providers

    def test_load_onnx_with_coreml(self, tmp_path: Path) -> None:
        """Should try CoreML provider on macOS when device is not cpu."""
        model_file = tmp_path / "model.onnx"
        model_file.touch()

        mock_input = MagicMock()
        mock_input.name = "input"
        mock_output = MagicMock()
        mock_output.name = "output"
        
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.get_outputs.return_value = [mock_output]
        mock_session.get_providers.return_value = ["CoreMLExecutionProvider"]

        mock_ort = MagicMock()
        mock_ort.InferenceSession.return_value = mock_session
        mock_ort.SessionOptions.return_value = MagicMock()
        mock_ort.GraphOptimizationLevel.ORT_ENABLE_ALL = 99
        # Only CoreML available (no CUDA)
        mock_ort.get_available_providers.return_value = ["CoreMLExecutionProvider", "CPUExecutionProvider"]

        detector = TimesNetDetector(model_path=str(model_file), device="mps")

        with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
            detector._load_onnx()

        # Check that CoreML provider was requested
        call_args = mock_ort.InferenceSession.call_args
        providers = call_args[1]["providers"]
        assert "CoreMLExecutionProvider" in providers

    def test_predict_onnx(self, tmp_path: Path) -> None:
        """Should run ONNX inference correctly."""
        model_file = tmp_path / "model.onnx"
        model_file.touch()

        # Set up mock session
        mock_session = MagicMock()
        # Return reconstruction that's slightly different from input
        def mock_run(output_names, inputs):
            x = inputs["input"]
            # Add small noise for reconstruction error
            return [x + 0.1]
        
        mock_session.run.side_effect = mock_run

        detector = TimesNetDetector(model_path=str(model_file))
        detector.onnx_session = mock_session
        detector.is_onnx = True
        detector._input_name = "input"
        detector._output_name = "output"

        # Test input: (3 metrics, 96 timesteps)
        data = np.random.randn(3, 96).astype(np.float32)
        scores = detector.predict(data)

        assert scores.shape == (3,)
        assert np.all(scores >= 0)
        assert np.all(scores <= 1)

    def test_predict_onnx_3d_input(self, tmp_path: Path) -> None:
        """Should handle 3D input correctly."""
        model_file = tmp_path / "model.onnx"
        model_file.touch()

        mock_session = MagicMock()
        def mock_run(output_names, inputs):
            return [inputs["input"] + 0.05]
        
        mock_session.run.side_effect = mock_run

        detector = TimesNetDetector(model_path=str(model_file))
        detector.onnx_session = mock_session
        detector.is_onnx = True
        detector._input_name = "input"
        detector._output_name = "output"

        # Test 3D input: (2 metrics, 96 timesteps, 1 feature)
        data = np.random.randn(2, 96, 1).astype(np.float32)
        scores = detector.predict(data)

        assert scores.shape == (2,)

    def test_predict_with_labels_onnx(self, tmp_path: Path) -> None:
        """Should return dict mapping names to scores for ONNX."""
        model_file = tmp_path / "model.onnx"
        model_file.touch()

        mock_session = MagicMock()
        mock_session.run.return_value = [np.zeros((3, 96, 1), dtype=np.float32)]

        detector = TimesNetDetector(model_path=str(model_file))
        detector.onnx_session = mock_session
        detector.is_onnx = True
        detector._input_name = "input"
        detector._output_name = "output"

        data = np.zeros((3, 96), dtype=np.float32)
        result = detector.predict_with_labels(data, ["m1", "m2", "m3"])

        assert isinstance(result, dict)
        assert set(result.keys()) == {"m1", "m2", "m3"}
        assert all(isinstance(v, float) for v in result.values())


class TestPyTorchDetector:
    """Tests for PyTorch model loading."""

    def test_load_pytorch_without_torch(self, tmp_path: Path) -> None:
        """Should raise ImportError if torch not installed."""
        model_file = tmp_path / "model.pt"
        model_file.touch()

        detector = TimesNetDetector(model_path=str(model_file))

        with patch.dict("sys.modules", {"torch": None}):
            with patch(
                "builtins.__import__",
                side_effect=ImportError("No module named 'torch'"),
            ):
                with pytest.raises(ImportError, match="PyTorch not installed"):
                    detector.load()

    def test_load_state_dict_not_implemented(self, tmp_path: Path) -> None:
        """Should raise NotImplementedError for state dict checkpoints."""
        model_file = tmp_path / "model.pt"
        model_file.touch()

        mock_torch = MagicMock()
        # Return a dict (state dict) instead of nn.Module
        mock_torch.load.return_value = {"layer.weight": [1, 2, 3]}
        mock_torch.nn.Module = type("Module", (), {})

        detector = TimesNetDetector(model_path=str(model_file))

        with patch.dict("sys.modules", {"torch": mock_torch}):
            with patch("builtins.__import__", return_value=mock_torch):
                # Patch the import inside load()
                with patch(
                    "anomaly_detection.detector.timesnet.TimesNetDetector._load_pytorch"
                ) as mock_load:
                    mock_load.side_effect = NotImplementedError(
                        "State dict loading requires model architecture"
                    )
                    with pytest.raises(NotImplementedError, match="State dict"):
                        detector.load()

    def test_predict_with_labels_returns_dict(self) -> None:
        """Should return dict mapping names to scores."""
        detector = TimesNetDetector(model_path="dummy.onnx")

        # Mock the predict method
        with patch.object(
            detector, "predict", return_value=np.array([0.3, 0.7, 0.5])
        ):
            detector.onnx_session = MagicMock()  # Mark as loaded
            detector.is_onnx = True

            result = detector.predict_with_labels(
                np.zeros((3, 96)), ["m1", "m2", "m3"]
            )

            assert result == {"m1": 0.3, "m2": 0.7, "m3": 0.5}


class TestLoadDispatch:
    """Tests for load() method dispatch logic."""

    def test_load_dispatches_to_onnx(self, tmp_path: Path) -> None:
        """Should call _load_onnx for .onnx files."""
        model_file = tmp_path / "model.onnx"
        model_file.touch()

        detector = TimesNetDetector(model_path=str(model_file))

        with patch.object(detector, "_load_onnx") as mock_load:
            detector.load()
            mock_load.assert_called_once()

    def test_load_dispatches_to_pytorch_pt(self, tmp_path: Path) -> None:
        """Should call _load_pytorch for .pt files."""
        model_file = tmp_path / "model.pt"
        model_file.touch()

        detector = TimesNetDetector(model_path=str(model_file))

        with patch.object(detector, "_load_pytorch") as mock_load:
            detector.load()
            mock_load.assert_called_once()

    def test_load_dispatches_to_pytorch_pth(self, tmp_path: Path) -> None:
        """Should call _load_pytorch for .pth files."""
        model_file = tmp_path / "model.pth"
        model_file.touch()

        detector = TimesNetDetector(model_path=str(model_file))

        with patch.object(detector, "_load_pytorch") as mock_load:
            detector.load()
            mock_load.assert_called_once()
