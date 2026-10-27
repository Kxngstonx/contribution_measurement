"""Frozen public encoders as drop-in replacements for the pooled anchor.

An `Encoder` is deliberately narrow: a `transform` for building a torchvision
dataset, and an `encode` that turns a batch of transformed images into
embeddings.  Nothing here ever sees more than one party's own images, which
is what lets `embeddings.py` extract a client's or the server's rows in
isolation.
"""

from __future__ import annotations

import abc

import torch


# Fixed kernel-temperature constants, analogous to the anchor path's existing
# hardcoded default of 20.0 -- no dynamic per-setting tau* search.
DEFAULT_KERNEL_TEMPERATURE: dict[str, float] = {
    "anchor": 20.0,
    "dinov2_vitb14": 160.0,
}


def _pil_to_rgb(image):
    """Picklable grayscale-to-RGB channel replication for DataLoader workers."""

    return image.convert("RGB")


class Encoder(abc.ABC):
    """Everything embedding extraction needs from a frozen encoder."""

    name: str
    embedding_dim: int
    transform: object

    @abc.abstractmethod
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Map a batch of already-transformed, device-resident images to embeddings."""

    @abc.abstractmethod
    def eval(self) -> "Encoder":
        """Switch to inference mode (no-op for encoders without training state)."""


class Dinov2Vitb14Encoder(Encoder):
    """timm `vit_base_patch14_dinov2.lvd142m`, frozen, pinned to 224px RGB."""

    def __init__(self, *, device: str | torch.device, image_size: int = 224):
        import timm
        from torchvision import transforms

        self.name = "dinov2_vitb14"
        self.device = torch.device(device)
        # Native input is 518px; passing img_size at creation time makes
        # timm interpolate the pretrained positional embeddings down to
        # image_size, rather than leaving a 518px-shaped pos_embed that
        # mismatches a 224px patch grid at forward time.
        self.model = timm.create_model(
            "vit_base_patch14_dinov2.lvd142m",
            pretrained=True,
            num_classes=0,
            img_size=image_size,
        )
        self.model.to(self.device)
        self.model.eval()
        self.embedding_dim = self.model.num_features

        config = self.model.pretrained_cfg
        mean = config.get("mean", (0.485, 0.456, 0.406))
        std = config.get("std", (0.229, 0.224, 0.225))
        self.transform = transforms.Compose(
            [
                _pil_to_rgb,
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(images.to(self.device))

    def eval(self) -> "Dinov2Vitb14Encoder":
        self.model.eval()
        return self


class StubEncoder(Encoder):
    """Deterministic, network-free encoder for dependency-free tests.

    A fixed random linear projection from flattened pixels to
    `embedding_dim`, seeded for reproducibility. Never touches `timm` or the
    network, so `scripts/run_tests.py` can exercise the embedding-extraction
    plumbing without any real encoder installed.
    """

    def __init__(
        self,
        *,
        input_shape: tuple[int, int, int],
        embedding_dim: int = 8,
        seed: int = 0,
    ):
        channels, height, width = input_shape
        generator = torch.Generator().manual_seed(seed)
        self.weight = torch.randn(
            embedding_dim, channels * height * width, generator=generator
        )
        self.name = "stub"
        self.embedding_dim = embedding_dim
        self.transform = None

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return images.flatten(1).float() @ self.weight.t()

    def eval(self) -> "StubEncoder":
        return self


def load_encoder(name: str, *, device: str | torch.device) -> Encoder:
    if name == "dinov2_vitb14":
        return Dinov2Vitb14Encoder(device=device)
    raise ValueError(f"unknown encoder: {name}")
