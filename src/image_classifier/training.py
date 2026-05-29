from __future__ import annotations

from pathlib import Path

import torch
from torch import nn, optim

from .data import create_dataloaders
from .model import create_model


def _run_epoch(model: nn.Module, loader, criterion, optimizer=None) -> tuple[float, float]:
    # Determine whether this epoch is for training
    is_training = optimizer is not None
    model.train(is_training)
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for images, labels in loader:
        # Zero gradients before optimizer step when training
        if is_training:
            optimizer.zero_grad()

        # Forward pass and loss computation
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backprop and optimizer step in training mode
        if is_training:
            loss.backward()
            optimizer.step()

        # Compute predictions and aggregate metrics
        predictions = outputs.argmax(dim=1)
        running_loss += loss.item() * images.size(0)
        running_correct += (predictions == labels).sum().item()
        running_total += labels.size(0)

    # Compute average loss and accuracy for the epoch
    average_loss = running_loss / running_total
    accuracy = running_correct / running_total
    return average_loss, accuracy


def train_model(
    data_dir: Path,
    output_path: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    image_size: int,
) -> None:
    # Prepare dataloaders and model
    bundle = create_dataloaders(data_dir=data_dir, batch_size=batch_size, image_size=image_size)
    model = create_model(len(bundle.class_names))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop over epochs
    for epoch in range(epochs):
        train_loss, train_accuracy = _run_epoch(model, bundle.train_loader, criterion, optimizer)
        val_loss, val_accuracy = _run_epoch(model, bundle.val_loader, criterion)
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}"
        )

    # Ensure checkpoint directory exists and save model state
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "class_names": bundle.class_names,
            "image_size": image_size,
        },
        output_path,
    )
    print(f"Saved checkpoint to {output_path}")
