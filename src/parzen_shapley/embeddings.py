"""Per-party embedding extraction and an on-disk cache shared across settings.

`extract_client_embeddings` is the client-local analogue of
`parzen_shapley.pipeline.extract_embeddings`: it encodes exactly one
party's own rows (a client's, or the server's validation rows), never a
`DataLoader` pooled across multiple parties. `EmbeddingCache` lets the 27
settings, which share only 3 underlying datasets, avoid re-running the
encoder forward pass once per setting.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

import torch
from torch.utils.data import DataLoader, Dataset

from .anchors import IndexedRelabelledDataset
from .encoders import Encoder


def extract_client_embeddings(
    encoder: Encoder,
    dataset: Dataset,
    indices: Sequence[int],
    overrides: Mapping[int, int],
    *,
    batch_size: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode exactly one party's own rows.

    Call once per client with that client's own indices, and once for the
    server's own validation rows -- never pool multiple parties' raw images
    into a single `DataLoader`.
    """

    loader = DataLoader(
        IndexedRelabelledDataset(dataset, indices, overrides),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    encoder.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for images, batch_labels in loader:
            embeddings.append(encoder.encode(images).cpu())
            labels.append(batch_labels.clone())
    if not embeddings:
        return (
            torch.empty(0, encoder.embedding_dim),
            torch.empty(0, dtype=torch.long),
        )
    return torch.cat(embeddings), torch.cat(labels)


class EmbeddingCache(NamedTuple):
    """A full-split cache keyed by original torchvision row index."""

    indices: torch.Tensor  # (N,) sorted ascending, original dataset index
    embeddings: torch.Tensor  # (N, d)
    labels: torch.Tensor  # (N,) clean labels; mislabel overrides not applied

    def select(
        self, indices: Sequence[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return rows for exactly the requested indices, in that order."""

        position = {
            int(index): row for row, index in enumerate(self.indices.tolist())
        }
        rows = torch.as_tensor(
            [position[int(index)] for index in indices], dtype=torch.long
        )
        return self.embeddings[rows], self.labels[rows]


def apply_overrides(
    indices: Sequence[int],
    labels: torch.Tensor,
    overrides: Mapping[int, int],
) -> torch.Tensor:
    """Reproduce `IndexedRelabelledDataset`'s override semantics for labels
    pulled from a cache rather than a live dataset pass."""

    overridden = labels.clone()
    override_map = {
        int(index): int(label) for index, label in overrides.items()
    }
    for position, index in enumerate(indices):
        if int(index) in override_map:
            overridden[position] = override_map[int(index)]
    return overridden


def save_embedding_cache(
    path: str | Path,
    *,
    indices: torch.Tensor,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    order = torch.argsort(indices)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(
        {
            "indices": indices[order].cpu(),
            "embeddings": embeddings[order].cpu(),
            "labels": labels[order].cpu(),
        },
        temporary,
    )
    os.replace(temporary, destination)


def load_embedding_cache(path: str | Path) -> EmbeddingCache:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return EmbeddingCache(
        indices=payload["indices"],
        embeddings=payload["embeddings"],
        labels=payload["labels"],
    )
