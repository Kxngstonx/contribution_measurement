"""Task-model architectures used to generate frozen embeddings."""

from __future__ import annotations

import torch.nn.functional as F
from torch import nn


class CNNMnist(nn.Module):
    """MNIST/Fashion-MNIST architecture from the original experiment."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x, is_feat=False, preact=False):
        f1_pre = F.max_pool2d(self.conv1(x), 2)
        f1 = F.relu(f1_pre)
        f2_pre = F.max_pool2d(self.conv2_drop(self.conv2(f1)), 2)
        f2 = F.relu(f2_pre)
        x = f2.view(-1, f2.shape[1] * f2.shape[2] * f2.shape[3])
        f3_pre = self.fc1(x)
        f3 = F.relu(f3_pre)
        x = F.dropout(f3, training=self.training)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        if not is_feat:
            return output
        return ([f1_pre, f2_pre, f3_pre] if preact else [f1, f2, f3]), output


class CNNCifar(nn.Module):
    """CIFAR-10 architecture from the original experiment."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.conv5 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv7 = nn.Conv2d(128, 128, 1, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.conv8 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv9 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv10 = nn.Conv2d(256, 256, 1, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()
        self.conv11 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv12 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv13 = nn.Conv2d(512, 512, 1, padding=1)
        self.pool5 = nn.MaxPool2d(2, 2, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU()
        self.fc14 = nn.Linear(512 * 4 * 4, 1024)
        self.drop1 = nn.Dropout()
        self.fc15 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.fc16 = nn.Linear(1024, 10)

    def forward(self, x, is_feat=False, preact=False):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        f1_pre = self.bn1(x)
        f1 = self.relu1(f1_pre)
        x = self.conv3(f1)
        x = self.conv4(x)
        x = self.pool2(x)
        f2_pre = self.bn2(x)
        f2 = self.relu2(f2_pre)
        x = self.conv5(f2)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        f3_pre = self.bn3(x)
        f3 = self.relu3(f3_pre)
        x = self.conv8(f3)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.pool4(x)
        f4_pre = self.bn4(x)
        f4 = self.relu4(f4_pre)
        x = self.conv11(f4)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.pool5(x)
        f5_pre = self.bn5(x)
        f5 = self.relu5(f5_pre)
        x = f5.view(-1, 512 * 4 * 4)
        x = F.relu(self.fc14(x))
        x = self.drop1(x)
        f6_pre = x
        f6 = F.relu(self.fc15(x))
        x = self.drop2(f6)
        output = F.log_softmax(self.fc16(x), dim=1)
        if not is_feat:
            return output
        pre = [f1_pre, f2_pre, f3_pre, f4_pre, f5_pre, f6_pre]
        post = [f1, f2, f3, f4, f5, f6]
        return (pre if preact else post), output


def build_task_model(dataset: str) -> nn.Module:
    if dataset == "cifar":
        return CNNCifar()
    if dataset in {"mnist", "fmnist"}:
        return CNNMnist()
    raise ValueError(f"unknown dataset: {dataset}")
