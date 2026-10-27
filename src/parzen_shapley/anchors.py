"""Deterministic training helpers for frozen task-model anchors."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class IndexedRelabelledDataset(Dataset):
    """View selected dataset rows with optional label overrides."""

    def __init__(
        self,
        dataset: Dataset,
        indices: Sequence[int],
        overrides: Mapping[int | str, int],
    ):
        self.dataset = dataset
        self.indices = [int(index) for index in indices]
        self.overrides = {
            int(index): int(label) for index, label in overrides.items()
        }

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        index = self.indices[position]
        image, label = self.dataset[index]
        return image, self.overrides.get(index, int(label))


def stable_int(key: str, seed: int) -> int:
    """Derive the historical setting-specific DataLoader seed."""

    digest = hashlib.sha256(key.encode()).digest()
    return int(seed) + int.from_bytes(digest[:4], "big")


def configure_determinism(seed: int) -> None:
    """Configure the same model-initialization determinism as the sweep."""

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def train_anchor(
    model: torch.nn.Module,
    dataset: Dataset,
    indices: Sequence[int],
    overrides: Mapping[int | str, int],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    momentum: float,
    device: str | torch.device,
    loader_seed: int,
    class_weights: Sequence[float] | None = None,
    max_steps: int | None = None,
) -> torch.nn.Module:
    """Train one task model on the union of a setting's client rows.

    `class_weights` (per class) turns the loss into `F.nll_loss`'s weighted
    mean; None keeps the plain mean. `max_steps` stops after that many
    optimizer steps (the first batches of the shuffled loader) even if
    `epochs` is not finished; None trains every epoch in full.
    """

    if epochs < 1:
        raise ValueError("epochs must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if len(indices) == 0:
        raise ValueError("indices must not be empty")
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be positive")
    generator = torch.Generator().manual_seed(loader_seed)
    loader = DataLoader(
        IndexedRelabelledDataset(dataset, indices, overrides),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    target_device = torch.device(device)
    model.to(target_device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
    )
    weight = (
        None if class_weights is None
        else torch.as_tensor(class_weights, dtype=torch.float32, device=target_device)
    )
    steps = 0
    for _ in range(epochs):
        model.train()
        for images, labels in loader:
            if max_steps is not None and steps >= max_steps:
                return model
            images = images.to(target_device)
            labels = labels.to(target_device)
            optimizer.zero_grad()
            loss = F.nll_loss(model(images), labels, weight=weight)
            loss.backward()
            optimizer.step()
            steps += 1
    return model


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Copy a model state dict to CPU for portable serialization."""

    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
    }


def save_checkpoint(model: torch.nn.Module, path: str | Path) -> None:
    """Atomically save the CPU state dict expected by downstream scripts."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(cpu_state_dict(model), temporary)
    os.replace(temporary, destination)
