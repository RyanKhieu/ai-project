from __future__ import annotations

# CLI entrypoint to train the image classifier

import argparse
from pathlib import Path

import torch

from src.image_classifier.training import train_model


def parse_args() -> argparse.Namespace:
    # Build argument parser for training configuration
    parser = argparse.ArgumentParser(description="Train an image recognition classifier.")
    # Dataset root containing class subfolders
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"), help="Folder containing class subfolders")
    # Output checkpoint path
    parser.add_argument("--output", type=Path, default=Path("checkpoints/flower_classifier.pt"), help="Where to save the checkpoint")
    # Number of epochs to train
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    # Mini-batch size for DataLoader
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    # Optimizer learning rate
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    # Input image size for transforms
    parser.add_argument("--image-size", type=int, default=224, help="Input image size")
    return parser.parse_args()


def main() -> None:
    # Parse CLI options
    args = parse_args()

    # Seed PyTorch RNG for reproducibility
    torch.manual_seed(42)

    # Ensure output directory exists for checkpoint
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Run the training routine with provided configuration
    train_model(
        data_dir=args.data_dir,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        image_size=args.image_size,
    )


if __name__ == "__main__":
    main()
