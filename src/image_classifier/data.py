from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


@dataclass(frozen=True)
class DatasetBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    class_names: list[str]

# Data transforms and dataloader construction utilities


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    # Normalization for ImageNet-pretrained models
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            normalize,
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, val_transform


def create_dataloaders(data_dir: Path, batch_size: int, image_size: int, val_split: float = 0.2) -> DatasetBundle:
    # Build train/validation transforms and a dataset for splitting
    train_transform, val_transform = build_transforms(image_size)
    dataset_for_split = datasets.ImageFolder(data_dir)

    # Ensure dataset is large enough to split
    if len(dataset_for_split) < 2:
        raise ValueError("The dataset needs at least two images to create a train/validation split.")

    # Compute train/validation split sizes and deterministic shuffled indices
    val_size = max(1, int(len(dataset_for_split) * val_split))
    train_size = len(dataset_for_split) - val_size
    generator = torch.Generator().manual_seed(42)
    shuffled_indices = torch.randperm(len(dataset_for_split), generator=generator).tolist()
    train_indices = shuffled_indices[:train_size]
    val_indices = shuffled_indices[train_size:]

    # Create datasets with the appropriate transforms
    train_dataset = datasets.ImageFolder(data_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(data_dir, transform=val_transform)

    # Wrap subsets in DataLoader objects
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return DatasetBundle(train_loader=train_loader, val_loader=val_loader, class_names=dataset_for_split.classes)
