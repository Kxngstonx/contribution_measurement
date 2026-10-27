"""Federated-learning client model architectures (LeNet-5 scale).

These are deliberately separate from `models.py`'s `CNNMnist`/`CNNCifar`
(which stay untouched and keep serving the Parzen anchor/embedding
pipeline). Reusing `CNNCifar` (VGG13-style, ~14.5M params) for local FL
training made every client's normalized gradient direction converge to
near-identical (pairwise cosine similarity ~0.9999+ from round 0), which
collapses any cosine-based contribution signal -- confirmed empirically, not
a normalization artifact (cosine similarity is scale-invariant, so the
rescale target cannot explain it). The CGSV reference repo
(github.com/XinyiYS/Gradient-Driven-Rewards-to-Guarantee-Fairness-in-
Collaborative-Machine-Learning, utils/defined_models.py) uses much smaller
LeNet-5-scale nets for both MNIST and CIFAR-10; reproduced here verbatim.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import nn


class FLMnistNet(nn.Module):
    """CGSV reference repo's `CNN_Net`. Expects 32x32 single-channel input (pad
    28x28 MNIST/FashionMNIST with 2px zero-padding on every side first)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, 1)
        self.conv2 = nn.Conv2d(64, 16, 7, 1)
        self.fc1 = nn.Linear(4 * 4 * 16, 200)
        self.fc2 = nn.Linear(200, num_classes)

    def forward(self, x):
        x = x.view(-1, 1, 32, 32)
        x = F.tanh(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.tanh(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 16)
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class FLCifarNet(nn.Module):
    """CGSV reference repo's `CNN_Cifar10`. Standard 32x32x3 CIFAR-10 input."""

    def __init__(
        self, in_channels: int = 3, n_kernels: int = 16, num_classes: int = 10
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, n_kernels, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(n_kernels, 2 * n_kernels, 5)
        self.fc1 = nn.Linear(2 * n_kernels * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


def build_fl_model(dataset: str) -> nn.Module:
    if dataset == "cifar":
        return FLCifarNet()
    if dataset in {"mnist", "fmnist"}:
        return FLMnistNet()
    raise ValueError(f"unknown dataset: {dataset}")
