#!/usr/bin/env python3
"""
Export trained TimesNet model to ONNX format for faster inference.

Usage:
    # Export default model
    python scripts/export_onnx.py

    # Export specific model with custom settings
    python scripts/export_onnx.py --input models/timesnet.pt --seq_len 100 --features 38

The ONNX model is saved to models/timesnet.onnx
"""

import argparse
import sys
from pathlib import Path


def export_to_onnx(args: argparse.Namespace) -> Path:
    """Export PyTorch model to ONNX."""
    try:
        import torch
    except ImportError:
        print("❌ PyTorch not installed. Run: uv add torch")
        sys.exit(1)
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Model not found: {input_path}")
        print("\nTrain a model first:")
        print("  python scripts/train.py --dataset SMD")
        sys.exit(1)
    
    print(f"Loading model from: {input_path}")
    
    # Load the checkpoint (state dict)
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)
    
    # Need to reconstruct the model architecture
    # This is tricky because TSLib saves state_dict, not the full model
    # We need to import TSLib's TimesNet and create a compatible wrapper
    
    print(f"Checkpoint type: {type(checkpoint)}")
    
    if isinstance(checkpoint, dict):
        # It's a state dict - need to reconstruct model
        print("Checkpoint is a state dict. Reconstructing model architecture...")
        
        # Create a simple wrapper model that matches the anomaly detection interface
        model = create_timesnet_model(args)
        model.load_state_dict(checkpoint)
    else:
        # It's a full model
        model = checkpoint
    
    model.eval()
    
    # Create dummy input: (batch, seq_len, features)
    dummy_input = torch.randn(1, args.seq_len, args.features)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Exporting to ONNX...")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output path: {output_path}")
    
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    
    # Verify the export
    try:
        import onnx
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print(f"✓ ONNX model verified")
    except ImportError:
        print("⚠ Install onnx to verify: uv add onnx")
    except Exception as e:
        print(f"⚠ ONNX verification warning: {e}")
    
    # Test with onnxruntime
    try:
        import onnxruntime as ort
        import numpy as np
        
        session = ort.InferenceSession(str(output_path))
        test_input = np.random.randn(1, args.seq_len, args.features).astype(np.float32)
        result = session.run(None, {"input": test_input})
        print(f"✓ ONNX inference test passed. Output shape: {result[0].shape}")
    except ImportError:
        print("⚠ Install onnxruntime to test: uv add onnxruntime")
    except Exception as e:
        print(f"⚠ ONNX runtime test warning: {e}")
    
    print(f"\n✓ Model exported to: {output_path}")
    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"  File size: {file_size:.2f} MB")
    
    return output_path


def create_timesnet_model(args: argparse.Namespace):
    """
    Create a TimesNet model for anomaly detection.
    
    This is a simplified version that matches TSLib's architecture.
    """
    import torch
    import torch.nn as nn
    
    # Import TSLib if available
    tslib_path = find_tslib_path()
    if tslib_path:
        sys.path.insert(0, str(tslib_path))
        try:
            from models.TimesNet import Model
            
            # Create args namespace that TSLib expects
            class Args:
                task_name = "anomaly_detection"
                seq_len = args.seq_len
                label_len = 0
                pred_len = 0
                enc_in = args.features
                c_out = args.features
                d_model = args.d_model
                d_ff = args.d_ff
                e_layers = args.e_layers
                top_k = 5
                num_kernels = 6
                embed = "timeF"
                freq = "h"
                dropout = 0.1
            
            model = Model(Args())
            print(f"✓ Created TimesNet model using TSLib")
            return model
        except Exception as e:
            print(f"⚠ Could not use TSLib: {e}")
    
    # Fallback: create a simple reconstruction model
    print("Creating simplified TimesNet wrapper...")
    
    class SimpleTimesNet(nn.Module):
        """Simplified TimesNet for ONNX export."""
        
        def __init__(self, seq_len: int, features: int, d_model: int = 64):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(features, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            self.decoder = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, features),
            )
        
        def forward(self, x):
            # x: (batch, seq_len, features)
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded
    
    return SimpleTimesNet(args.seq_len, args.features, args.d_model)


def find_tslib_path() -> Path | None:
    """Find Time-Series-Library."""
    script_dir = Path(__file__).parent.parent.parent
    candidates = [
        script_dir / "Time-Series-Library",
        Path.home() / "Time-Series-Library",
    ]
    
    for path in candidates:
        if (path / "run.py").exists():
            return path
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Export TimesNet to ONNX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input",
        type=str,
        default="models/timesnet.pt",
        help="Input PyTorch model path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/timesnet.onnx",
        help="Output ONNX model path",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=100,
        help="Sequence length (must match training)",
    )
    parser.add_argument(
        "--features",
        type=int,
        default=38,
        help="Number of features (must match training)",
    )
    parser.add_argument(
        "--d_model",
        type=int,
        default=64,
        help="Model dimension (must match training)",
    )
    parser.add_argument(
        "--d_ff",
        type=int,
        default=64,
        help="Feed-forward dimension (must match training)",
    )
    parser.add_argument(
        "--e_layers",
        type=int,
        default=2,
        help="Number of encoder layers (must match training)",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("TimesNet ONNX Export")
    print("=" * 60)
    
    export_to_onnx(args)
    
    print("\n" + "=" * 60)
    print("Next step: Update detector to use ONNX")
    print("  Set MODEL_PATH=models/timesnet.onnx")
    print("=" * 60)


if __name__ == "__main__":
    main()
