from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from src.image_classifier.infer import predict_image


def parse_args() -> argparse.Namespace:
    # CLI argument parser for prediction
    parser = argparse.ArgumentParser(description="Predict the species of a flower image.")
    # Path to trained checkpoint to load
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/flower_classifier.pt"), help="Path to a trained checkpoint")
    # Path to input image to classify
    parser.add_argument("--image", type=Path, required=True, help="Path to the input image")
    return parser.parse_args()


def main() -> None:
    # Parse CLI options
    args = parse_args()

    # Load image and convert to RGB
    image = Image.open(args.image).convert("RGB")

    # Get prediction from the inference helper
    prediction = predict_image(image, args.checkpoint)

    # Print concise results to stdout
    print(f"Prediction: {prediction.label}")
    print(f"Confidence: {prediction.confidence:.4f}")
    print("Top predictions:")
    for label, score in prediction.top_k:
        # Print each top-k label and score
        print(f"  {label}: {score:.4f}")


if __name__ == "__main__":
    main()
