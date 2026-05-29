from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torchvision import models


@dataclass(frozen=True)
class ModelBundle:
    model: nn.Module
    class_names: list[str]

# Model creation and checkpoint loading utilities


def create_model(num_classes: int) -> nn.Module:
    # Load ResNet18 with default pretrained weights and replace final layer
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def load_checkpoint(checkpoint_path: str | Path) -> ModelBundle:
    # Load a saved checkpoint and reconstruct the model + class names
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    class_names = checkpoint["class_names"]
    model = create_model(len(class_names))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return ModelBundle(model=model, class_names=class_names)
