from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from .model import load_checkpoint


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    top_k: list[tuple[str, float]]


def build_inference_transform(image_size: int = 224) -> transforms.Compose:
    # Transform used at inference time to prepare images for the model
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def predict_image(image: Image.Image, checkpoint_path: Path, top_k: int = 3) -> Prediction:
    # Load model and class names from checkpoint
    bundle = load_checkpoint(checkpoint_path)

    # Prepare image tensor for model input
    transform = build_inference_transform()
    tensor = transform(image).unsqueeze(0)

    # Run model in eval mode without tracking gradients
    with torch.no_grad():
        logits = bundle.model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    # Extract top-k predictions and return a simple dataclass
    values, indices = torch.topk(probabilities, k=min(top_k, probabilities.numel()))
    top_predictions = [(bundle.class_names[index], float(value)) for value, index in zip(values.tolist(), indices.tolist())]
    best_label, best_confidence = top_predictions[0]
    return Prediction(label=best_label, confidence=best_confidence, top_k=top_predictions)
