"""Hyperparameters, datasets, stored client partitions and the server root."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets, transforms

from parzen_shapley.pipeline import dataset_objects

# Local-training hyperparameters, adopted from the CGSV reference repo
# (github.com/XinyiYS/Gradient-Driven-Rewards-to-Guarantee-Fairness-in-
# Collaborative-Machine-Learning: utils/arguments.py + main.py's per-dataset
# overrides); FMNIST reuses MNIST's. The batch size is 128 on every dataset so
# the 128-sample root is exactly one full-batch server step. `gamma_target` is
# the fixed norm every client pseudo-gradient is rescaled to before
# aggregation; `alpha` is the EMA factor of the contribution score.
TRAC_HYPERPARAMS = {
    "mnist": {
        "local_epochs": 3,
        "batch_size": 128,
        "learning_rate": 1e-3,
        "lr_decay_gamma": 0.955,
        "gamma_target": 0.5,
        "alpha": 0.95,
    },
    "fmnist": {
        "local_epochs": 3,
        "batch_size": 128,
        "learning_rate": 1e-3,
        "lr_decay_gamma": 0.955,
        "gamma_target": 0.5,
        "alpha": 0.95,
    },
    "cifar": {
        "local_epochs": 3,
        "batch_size": 128,
        "learning_rate": 0.015,
        "lr_decay_gamma": 0.977,
        "gamma_target": 0.15,
        "alpha": 0.95,
    },
}

ROUNDS = {"mnist": 30, "fmnist": 30, "cifar": 30}
ROOT_SIZE = 128
VALIDATION_SIZE = 2000
ROOT_SEED = 0


def fl_dataset_objects(dataset, data_root, *, download=False):
    """Same shape as `parzen_shapley.pipeline.dataset_objects`, but MNIST/
    FashionMNIST are zero-padded 28x28 -> 32x32 to match `FLMnistNet`'s
    hardcoded `view(-1, 1, 32, 32)` reshape. CIFAR-10 is already 32x32, so
    it's identical to `dataset_objects`."""

    if dataset == "cifar":
        return dataset_objects(dataset, data_root, download=download)
    transform = transforms.Compose(
        [
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    root = Path(data_root) / dataset
    dataset_class = datasets.FashionMNIST if dataset == "fmnist" else datasets.MNIST
    return (
        dataset_class(str(root), train=True, download=download, transform=transform),
        dataset_class(str(root), train=False, download=download, transform=transform),
    )


def load_partition(reference_dir, key):
    """(client_indices, overrides, client_sizes) of a stored setting."""

    reference = Path(reference_dir) / key
    groups = {
        int(client): [int(index) for index in indices]
        for client, indices in json.loads(
            (reference / "user_groups_0.json").read_text()
        ).items()
    }
    overrides = json.loads((reference / "mislabel_targets_0.json").read_text())
    n_players = len(groups)
    client_indices = [groups[player] for player in range(n_players)]
    client_sizes = [len(indices) for indices in client_indices]
    return client_indices, overrides, client_sizes


def client_label_arrays(train_dataset, client_indices):
    """Each client's (clean) training labels as numpy arrays."""

    targets = getattr(train_dataset, "targets", None)
    if targets is not None:
        targets = np.asarray(targets.tolist() if hasattr(targets, "tolist") else targets)
        return [targets[np.asarray(idx)] for idx in client_indices]
    return [
        np.asarray([int(train_dataset[i][1]) for i in idx]) for idx in client_indices
    ]


def expected_k(setting) -> int:
    """Number of contaminated clients a manifest setting declares."""

    metadata = setting["setting"]
    return int(metadata["num_mislabel"] or metadata["num_noniid"])


def cover_other_classes(rest, others, labels):
    """Make every class of `others` appear in `rest` at least once (same size).

    FLTrust's Case II fills the non-bias part of the root uniformly from the other
    classes. With few slots (26 for 9 classes at bias 0.8 x 128) a plain draw can
    miss a class, which confounds the bias with a missing class. For each absent
    class (in label order) the earliest member of `others` replaces the last-drawn
    member of the currently largest class in `rest`. A draw that already covers
    every class is returned unchanged; with fewer slots than classes it cannot be
    repaired and is returned unchanged too.
    """

    present = {int(labels[i]) for i in others}
    if len(rest) < len(present):
        return rest
    rest = list(rest)
    for label in sorted(present):
        counts = {}
        for i in rest:
            counts[int(labels[i])] = counts.get(int(labels[i]), 0) + 1
        if label in counts:
            continue
        largest = max(counts, key=lambda c: (counts[c], -c))
        drop = max(k for k, i in enumerate(rest) if int(labels[i]) == largest)
        rest.pop(drop)
        rest.append(next(i for i in others if int(labels[i]) == label))
    return rest


def server_and_eval_indices(
    test_dataset, *, validation_size=VALIDATION_SIZE, server_pc=ROOT_SIZE, seed=ROOT_SEED,
    bias_prob=None, bias_class=1, missing_classes=(),
):
    """Disjoint slices of the test split: the first `validation_size` indices
    reproduce the Parzen validation set (same randperm + seed as
    `parzen_shapley.pipeline.prepare_setting_utility`), the next `server_pc`
    are the TRAC server root.

    Skewed roots (server-data experiments), drawn from the same pool
    `perm[validation_size:]` in perm order:
    - `bias_prob` = FLTrust's Case II "bias probability": round(bias_prob *
      server_pc) examples of `bias_class`, the rest from the other classes
      uniformly at random (the earliest non-`bias_class` members of the pool),
      repaired by `cover_other_classes` so no other class is left out.
    - `missing_classes`: those labels are excluded from the pool before the
      plain draw.
    The validation set is never touched."""

    n = len(test_dataset)
    if validation_size + server_pc > n:
        raise ValueError(
            f"validation_size + server_pc = {validation_size + server_pc} "
            f"exceeds the test split ({n})"
        )
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    eval_idx = perm[:validation_size].tolist()
    missing = {int(c) for c in missing_classes}
    if bias_prob is None and not missing:
        server_idx = perm[validation_size : validation_size + server_pc].tolist()
        return server_idx, eval_idx

    labels = np.asarray(test_dataset.targets)
    pool = [i for i in perm[validation_size:].tolist() if int(labels[i]) not in missing]
    if bias_prob is not None:
        if not 0.0 <= bias_prob <= 1.0:
            raise ValueError(f"bias_prob must lie in [0, 1], got {bias_prob}")
        if bias_class in missing:
            raise ValueError("bias_class cannot be a missing class")
        n_bias = int(round(bias_prob * server_pc))
        biased = [i for i in pool if int(labels[i]) == bias_class][:n_bias]
        others = [i for i in pool if int(labels[i]) != bias_class]
        rest = others[: server_pc - n_bias]
        if len(biased) < n_bias or len(biased) + len(rest) < server_pc:
            raise ValueError("pool too small for the requested biased root")
        return biased + cover_other_classes(rest, others, labels), eval_idx
    return pool[:server_pc], eval_idx
