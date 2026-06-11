#!/usr/bin/env python3
"""
Train TimesNet for anomaly detection using Time-Series-Library.

Usage:
    # Train on SMD dataset (Server Machine Dataset)
    python scripts/train.py --dataset SMD

    # Train on PSM dataset (Pooled Server Metrics)
    python scripts/train.py --dataset PSM

    # Custom settings
    python scripts/train.py --dataset SMD --epochs 20 --seq_len 100

The trained model is saved to models/timesnet.pt
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


# Dataset configurations
DATASET_CONFIGS = {
    "SMD": {
        "enc_in": 38,  # 38 features
        "c_out": 38,
        "seq_len": 100,
        "anomaly_ratio": 0.5,
        "download_url": "https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2",
    },
    "PSM": {
        "enc_in": 25,  # 25 features
        "c_out": 25,
        "seq_len": 100,
        "anomaly_ratio": 1.0,
        "download_url": "https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2",
    },
    "MSL": {
        "enc_in": 55,  # 55 features (Mars Science Laboratory)
        "c_out": 55,
        "seq_len": 100,
        "anomaly_ratio": 1.0,
        "download_url": "https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2",
    },
    "SMAP": {
        "enc_in": 25,  # 25 features (Soil Moisture Active Passive)
        "c_out": 25,
        "seq_len": 100,
        "anomaly_ratio": 1.0,
        "download_url": "https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2",
    },
}


def find_tslib_path() -> Path:
    """Find Time-Series-Library in common locations."""
    # Check relative to this script
    script_dir = Path(__file__).parent.parent.parent
    candidates = [
        script_dir / "Time-Series-Library",
        Path.home() / "Time-Series-Library",
        Path("/workspace/Time-Series-Library"),
    ]
    
    for path in candidates:
        if (path / "run.py").exists():
            return path
    
    raise FileNotFoundError(
        "Time-Series-Library not found. Please clone it:\n"
        "git clone https://github.com/thuml/Time-Series-Library.git"
    )


def check_dataset(tslib_path: Path, dataset: str) -> bool:
    """Check if dataset exists."""
    dataset_path = tslib_path / "dataset" / dataset
    return dataset_path.exists() and any(dataset_path.iterdir())


def train(args: argparse.Namespace) -> Path:
    """Run training using TSLib."""
    tslib_path = find_tslib_path()
    config = DATASET_CONFIGS[args.dataset]
    
    # Check dataset
    if not check_dataset(tslib_path, args.dataset):
        print(f"\n❌ Dataset '{args.dataset}' not found at {tslib_path / 'dataset' / args.dataset}")
        print(f"\nDownload datasets from: {config['download_url']}")
        print(f"Then extract to: {tslib_path / 'dataset/'}")
        print("\nExpected structure:")
        print(f"  {tslib_path}/dataset/{args.dataset}/")
        print(f"    ├── train.npy")
        print(f"    ├── test.npy")
        print(f"    └── test_label.npy")
        sys.exit(1)
    
    print(f"✓ Dataset found: {args.dataset}")
    print(f"  Features: {config['enc_in']}")
    print(f"  Sequence length: {args.seq_len or config['seq_len']}")
    print(f"  Epochs: {args.epochs}")
    
    # Build training command
    cmd = [
        sys.executable, "run.py",
        "--task_name", "anomaly_detection",
        "--is_training", "1",
        "--root_path", f"./dataset/{args.dataset}",
        "--model_id", args.dataset,
        "--model", "TimesNet",
        "--data", args.dataset,
        "--features", "M",
        "--seq_len", str(args.seq_len or config["seq_len"]),
        "--pred_len", "0",
        "--d_model", str(args.d_model),
        "--d_ff", str(args.d_ff),
        "--e_layers", str(args.e_layers),
        "--enc_in", str(config["enc_in"]),
        "--c_out", str(config["c_out"]),
        "--top_k", "5",
        "--anomaly_ratio", str(config["anomaly_ratio"]),
        "--batch_size", str(args.batch_size),
        "--train_epochs", str(args.epochs),
    ]
    
    # Add GPU settings
    if args.no_gpu:
        cmd.extend(["--no_use_gpu"])
    else:
        cmd.extend(["--gpu_type", "mps" if sys.platform == "darwin" else "cuda"])
    
    print(f"\n🚀 Starting training...")
    print(f"   Command: {' '.join(cmd)}\n")
    
    # Run training
    result = subprocess.run(cmd, cwd=tslib_path)
    
    if result.returncode != 0:
        print("❌ Training failed")
        sys.exit(1)
    
    # Find the checkpoint
    checkpoint_pattern = f"anomaly_detection_{args.dataset}_TimesNet_{args.dataset}_*"
    checkpoints_dir = tslib_path / "checkpoints"
    
    matching = list(checkpoints_dir.glob(f"{checkpoint_pattern}/checkpoint.pth"))
    if not matching:
        print(f"❌ Checkpoint not found in {checkpoints_dir}")
        sys.exit(1)
    
    # Use the most recent
    checkpoint_path = max(matching, key=lambda p: p.stat().st_mtime)
    print(f"\n✓ Training complete. Checkpoint: {checkpoint_path}")
    
    return checkpoint_path


def copy_checkpoint(checkpoint_path: Path, output_dir: Path) -> Path:
    """Copy checkpoint to output directory."""
    import shutil
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "timesnet.pt"
    
    shutil.copy2(checkpoint_path, output_path)
    print(f"✓ Model saved to: {output_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Train TimesNet for anomaly detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_CONFIGS.keys()),
        default="SMD",
        help="Dataset to train on (default: SMD)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=None,
        help="Sequence length (default: dataset-specific)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size (default: 128)",
    )
    parser.add_argument(
        "--d_model",
        type=int,
        default=64,
        help="Model dimension (default: 64)",
    )
    parser.add_argument(
        "--d_ff",
        type=int,
        default=64,
        help="Feed-forward dimension (default: 64)",
    )
    parser.add_argument(
        "--e_layers",
        type=int,
        default=2,
        help="Number of encoder layers (default: 2)",
    )
    parser.add_argument(
        "--no_gpu",
        action="store_true",
        help="Disable GPU (use CPU only)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(__file__).parent.parent / "models",
        help="Output directory for trained model",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("TimesNet Anomaly Detection Training")
    print("=" * 60)
    
    checkpoint_path = train(args)
    copy_checkpoint(checkpoint_path, args.output_dir)
    
    print("\n" + "=" * 60)
    print("Next step: Export to ONNX for faster inference")
    print("  python scripts/export_onnx.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
