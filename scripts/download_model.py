#!/usr/bin/env python3
"""
Download or train a TimesNet model for anomaly detection.

Usage:
    # Download pre-trained checkpoint (if available)
    python scripts/download_model.py --download
    
    # Train on SMD dataset (from Time-Series-Library)
    python scripts/download_model.py --train --dataset SMD
    
    # Train on your own Prometheus data
    python scripts/download_model.py --train --prometheus-url http://localhost:9090
"""

import argparse
import sys
from pathlib import Path


def download_pretrained(output_path: Path) -> None:
    """
    Download a pre-trained TimesNet checkpoint.
    
    Note: As of now, there's no official pre-trained TimesNet for anomaly detection.
    You'll need to train your own using Time-Series-Library.
    """
    print("⚠️  No official pre-trained TimesNet anomaly detection model available.")
    print()
    print("Options:")
    print("  1. Train on SMD dataset: python scripts/download_model.py --train --dataset SMD")
    print("  2. Train on your data:   python scripts/download_model.py --train --prometheus-url URL")
    print("  3. Use statistical detector: DETECTOR_TYPE=statistical make run")
    print()
    print("See: https://github.com/thuml/Time-Series-Library for training details.")
    sys.exit(1)


def train_on_dataset(dataset: str, output_path: Path) -> None:
    """
    Train TimesNet on a standard benchmark dataset using Time-Series-Library.
    """
    try:
        import torch
    except ImportError:
        print("❌ PyTorch not installed. Run: uv add torch")
        sys.exit(1)
    
    print(f"Training TimesNet on {dataset} dataset...")
    print()
    print("This requires Time-Series-Library. Steps:")
    print()
    print("  1. Clone Time-Series-Library:")
    print("     git clone https://github.com/thuml/Time-Series-Library.git")
    print()
    print("  2. Download SMD dataset:")
    print("     # Download from https://github.com/NetManAIOps/OmniAnomaly")
    print("     # Place in Time-Series-Library/dataset/SMD/")
    print()
    print("  3. Run training:")
    print("     cd Time-Series-Library")
    print("     python -u run.py \\")
    print("       --task_name anomaly_detection \\")
    print("       --is_training 1 \\")
    print("       --model_id SMD_TimesNet \\")
    print("       --model TimesNet \\")
    print("       --data SMD \\")
    print("       --root_path ./dataset/SMD \\")
    print("       --seq_len 100 \\")
    print("       --pred_len 0 \\")
    print("       --d_model 64 \\")
    print("       --d_ff 64 \\")
    print("       --e_layers 2 \\")
    print("       --top_k 3 \\")
    print("       --batch_size 128 \\")
    print("       --train_epochs 3")
    print()
    print("  4. Copy the model:")
    print(f"     cp Time-Series-Library/checkpoints/SMD_TimesNet/checkpoint.pth {output_path}")
    print()
    print("  5. Run with TimesNet:")
    print("     make run")
    

def train_on_prometheus(prometheus_url: str, output_path: Path) -> None:
    """
    Train TimesNet on your own Prometheus metrics.
    """
    print(f"Training on Prometheus data from {prometheus_url}...")
    print()
    print("This is a custom training workflow. Steps:")
    print()
    print("  1. Export metrics to CSV:")
    print("     python scripts/export_prometheus.py --url", prometheus_url)
    print()
    print("  2. Format as Time-Series-Library dataset")
    print()
    print("  3. Train (see --dataset option)")
    print()
    print("For production, consider using the statistical detector while")
    print("collecting enough data for training:")
    print("  DETECTOR_TYPE=statistical make run")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download or train TimesNet model")
    parser.add_argument("--download", action="store_true", help="Download pre-trained model")
    parser.add_argument("--train", action="store_true", help="Train a new model")
    parser.add_argument("--dataset", type=str, default="SMD", help="Dataset name for training")
    parser.add_argument("--prometheus-url", type=str, help="Prometheus URL for custom training")
    parser.add_argument("--output", type=str, default="models/timesnet.pt", help="Output path")
    
    args = parser.parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if args.download:
        download_pretrained(output_path)
    elif args.train:
        if args.prometheus_url:
            train_on_prometheus(args.prometheus_url, output_path)
        else:
            train_on_dataset(args.dataset, output_path)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
