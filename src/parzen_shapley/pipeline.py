"""Setting catalog, datasets, anchors and the per-setting Parzen utility.

The pieces of the Parzen-Shapley pipeline that both the batch scripts
(`scripts/train_anchors.py`, `scripts/build_parzen_games.py`,
`scripts/estimate_sampled.py`) and the demo program share.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .anchors import (
    IndexedRelabelledDataset,
    configure_determinism,
    save_checkpoint,
    stable_int,
    train_anchor,
)
from .embeddings import apply_overrides, extract_client_embeddings, load_embedding_cache
from .models import build_task_model
from .utility import assemble_parzen_utility, client_local_statistics, make_parzen_utility

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_SPACE = Path(os.environ.get("SPACE_ROOT", str(PROJECT.parent / "SPACE")))
DEFAULT_DATA_ROOT = Path(os.environ.get("DATA_ROOT", str(DEFAULT_SPACE / "data")))
DEFAULT_REFERENCE = PROJECT / "assets" / "reference"
DEFAULT_MANIFEST = PROJECT / "configs" / "space_exact_gt_manifest.json"

# Anchor training protocol (frozen since the GT sweep): 5 epochs of SGD on the
# union of the clients' data with the stored label overrides applied.
ANCHOR_EPOCHS = 5
ANCHOR_BATCH_SIZE = 128
ANCHOR_LEARNING_RATE = 0.01
ANCHOR_MOMENTUM = 0.5


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------


def scenario(setting):
    metadata = setting["setting"]
    if metadata["num_mislabel"]:
        return f"mislabel_{metadata['num_mislabel']}"
    if metadata["num_noniid"]:
        return f"noniid_{metadata['num_noniid']}"
    return "iid"


def selected_settings(manifest):
    """The 27 paper-scale settings (MNIST-10, FMNIST-10, CIFAR-5) in dataset /
    iid-mislabel-noniid / severity order."""

    expected = {"mnist": 10, "fmnist": 10, "cifar": 5}
    rows = [
        setting
        for setting in manifest["settings"]
        if setting["setting"]["num_users"] == expected.get(setting["dataset"])
    ]
    family_order = {"iid": 0, "mislabel": 1, "noniid": 2}

    def order(setting):
        label = scenario(setting)
        family = label.split("_", 1)[0]
        severity = int(label.split("_")[1]) if "_" in label else 0
        return (
            ["mnist", "fmnist", "cifar"].index(setting["dataset"]),
            family_order[family],
            severity,
        )

    return sorted(rows, key=order)


def load_settings(manifest_path=DEFAULT_MANIFEST):
    return selected_settings(json.loads(Path(manifest_path).read_text()))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def setting_inputs(setting, reference_root: Path):
    """(groups, overrides, all_indices, verified_hashes) of a setting's stored
    partition, verified against the manifest's sha256 when present."""

    reference = Path(reference_root) / setting["key"]
    groups_path = reference / "user_groups_0.json"
    overrides_path = reference / "mislabel_targets_0.json"
    missing = [str(path) for path in (groups_path, overrides_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing stored client partition files: " + ", ".join(missing)
        )
    verified_hashes = {}
    expected_hashes = setting.get("sha256", {})
    for input_path in (groups_path, overrides_path):
        actual = file_sha256(input_path)
        expected = expected_hashes.get(input_path.name)
        if expected is not None and actual != expected:
            raise ValueError(
                f"partition hash mismatch for {input_path}: "
                f"expected {expected}, got {actual}"
            )
        verified_hashes[input_path.name] = actual
    groups = {
        int(client): [int(index) for index in indices]
        for client, indices in json.loads(groups_path.read_text()).items()
    }
    expected = list(range(len(groups)))
    if sorted(groups) != expected:
        raise ValueError(
            f"client ids for {setting['key']} must be contiguous: "
            f"expected {expected}, got {sorted(groups)}"
        )
    overrides = json.loads(overrides_path.read_text())
    all_indices = sorted({index for client in expected for index in groups[client]})
    if not all_indices:
        raise ValueError(f"setting {setting['key']} has no training rows")
    return groups, overrides, all_indices, verified_hashes


# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------


def dataset_objects(dataset, data_root, *, download=False):
    """(train, test) torchvision datasets with the anchor models' transforms."""

    if dataset == "cifar":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        root = Path(data_root) / "cifar"
        dataset_class = datasets.CIFAR10
    else:
        transform = transforms.Compose(
            [
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


def dataset_objects_for_encoder(dataset, data_root, encoder, *, download=False):
    """Same shape as `dataset_objects`, but transformed for a public encoder
    rather than the anchor's hardcoded MNIST/CIFAR transforms."""

    if dataset == "cifar":
        root = Path(data_root) / "cifar"
        dataset_class = datasets.CIFAR10
    else:
        root = Path(data_root) / dataset
        dataset_class = datasets.FashionMNIST if dataset == "fmnist" else datasets.MNIST
    return (
        dataset_class(str(root), train=True, download=download, transform=encoder.transform),
        dataset_class(str(root), train=False, download=download, transform=encoder.transform),
    )


def encoder_cache_paths(cache_dir, encoder_name, dataset):
    """Paths to the (encoder, dataset, split) embedding cache, or None if
    caching is disabled (`cache_dir` falsy)."""

    if not cache_dir:
        return None
    base = Path(cache_dir)
    return {
        "train": base / f"{encoder_name}__{dataset}__train.pt",
        "test": base / f"{encoder_name}__{dataset}__test.pt",
    }


# --------------------------------------------------------------------------
# Anchors and embeddings
# --------------------------------------------------------------------------


def train_setting_anchor(setting, train_dataset, checkpoint, *, device, seed=0):
    """Train and save the frozen task-model anchor of `setting` with the
    project's fixed protocol; returns the trained model."""

    _groups, overrides, all_indices, _hashes = setting_inputs(setting, DEFAULT_REFERENCE)
    configure_determinism(seed)
    model = build_task_model(setting["dataset"])
    model = train_anchor(
        model,
        train_dataset,
        all_indices,
        overrides,
        epochs=ANCHOR_EPOCHS,
        batch_size=ANCHOR_BATCH_SIZE,
        learning_rate=ANCHOR_LEARNING_RATE,
        momentum=ANCHOR_MOMENTUM,
        device=torch.device(device),
        loader_seed=stable_int(setting["key"], seed),
    )
    save_checkpoint(model, checkpoint)
    return model


def extract_embeddings(model, loader, device):
    embeddings = []
    labels = []
    model.eval()
    with torch.no_grad():
        for images, batch_labels in loader:
            features, _ = model(images.to(device), is_feat=True, preact=False)
            embeddings.append(features[-1].flatten(1).cpu())
            labels.append(batch_labels.clone())
    return torch.cat(embeddings), torch.cat(labels)


def prepare_setting_utility(
    setting, train_dataset, test_dataset, args, device, *, encoder=None
):
    """Construct one on-demand Parzen game without enumerating coalitions.

    `args` carries `reference_dir, anchor_dir, seed, validation_size,
    feature_batch_size, parzen_batch_size, kernel_temperature, encoder`
    (plus `embedding_cache_dir` for a public encoder). `args.encoder ==
    "anchor"` (the default) keeps the original pooled path byte-for-byte
    unchanged. Any other `args.encoder` takes the FL-shaped path: `encoder`
    (already instantiated once by the caller) is used to extract each
    client's -- and the validation set's -- embeddings independently, and
    the coalition utility is assembled from those per-client statistics via
    `client_local_statistics`/`assemble_parzen_utility` rather than a single
    pooled pass.
    """

    key = setting["key"]
    dataset = setting["dataset"]
    reference = Path(args.reference_dir) / key
    groups = {
        int(client): [int(index) for index in indices]
        for client, indices in json.loads((reference / "user_groups_0.json").read_text()).items()
    }
    overrides = json.loads((reference / "mislabel_targets_0.json").read_text())
    n_players = len(groups)
    all_indices = sorted({index for indices in groups.values() for index in indices})
    validation_generator = torch.Generator().manual_seed(args.seed)
    validation_indices = torch.randperm(
        len(test_dataset), generator=validation_generator
    )[: args.validation_size].tolist()

    if args.encoder == "anchor":
        positions = {global_index: row for row, global_index in enumerate(all_indices)}
        client_rows = [
            np.asarray([positions[index] for index in groups[player]], dtype=np.int64)
            for player in range(n_players)
        ]
        train_loader = DataLoader(
            IndexedRelabelledDataset(train_dataset, all_indices, overrides),
            batch_size=args.feature_batch_size,
            shuffle=False,
            num_workers=0,
        )
        validation_loader = DataLoader(
            IndexedRelabelledDataset(test_dataset, validation_indices, {}),
            batch_size=args.feature_batch_size,
            shuffle=False,
            num_workers=0,
        )
        anchor_path = Path(args.anchor_dir) / f"{key}.pt"
        if not anchor_path.is_file():
            raise FileNotFoundError(anchor_path)
        model = build_task_model(dataset)
        model.load_state_dict(
            torch.load(anchor_path, map_location="cpu", weights_only=True)
        )
        model.to(device)
        embeddings, labels = extract_embeddings(model, train_loader, device)
        validation_embeddings, validation_labels = extract_embeddings(
            model, validation_loader, device
        )
        parzen = make_parzen_utility(
            embeddings,
            labels,
            client_rows,
            validation_embeddings,
            validation_labels,
            kernel_temperature=args.kernel_temperature,
            batch_size=args.parzen_batch_size,
            device=device,
        )
        anchor_checkpoint = str(anchor_path.resolve())
        embedding_source = f"anchor:{anchor_checkpoint}"
    else:
        if encoder is None:
            raise ValueError("encoder must be provided for non-anchor runs")
        cache_paths = encoder_cache_paths(args.embedding_cache_dir, args.encoder, dataset)
        train_cache = (
            load_embedding_cache(cache_paths["train"])
            if cache_paths and cache_paths["train"].is_file()
            else None
        )
        test_cache = (
            load_embedding_cache(cache_paths["test"])
            if cache_paths and cache_paths["test"].is_file()
            else None
        )

        if test_cache is not None:
            raw_validation_embeddings, validation_labels = test_cache.select(validation_indices)
        else:
            raw_validation_embeddings, validation_labels = extract_client_embeddings(
                encoder,
                test_dataset,
                validation_indices,
                {},
                batch_size=args.feature_batch_size,
                device=device,
            )
        # Server-side broadcast prep: normalize once, share across clients.
        validation_broadcast = F.normalize(
            raw_validation_embeddings.detach().float(), p=2, dim=-1
        ).to(device)
        classes = torch.unique(validation_labels.to(device), sorted=True)

        statistics = []
        for player in range(n_players):
            client_indices = groups[player]
            if train_cache is not None:
                client_embeddings, clean_labels = train_cache.select(client_indices)
                client_labels = apply_overrides(client_indices, clean_labels, overrides)
            else:
                client_embeddings, client_labels = extract_client_embeddings(
                    encoder,
                    train_dataset,
                    client_indices,
                    overrides,
                    batch_size=args.feature_batch_size,
                    device=device,
                )
            statistics.append(
                client_local_statistics(
                    client_embeddings,
                    client_labels,
                    validation_broadcast,
                    classes,
                    kernel_temperature=args.kernel_temperature,
                    batch_size=args.parzen_batch_size,
                    device=device,
                )
            )
        parzen = assemble_parzen_utility(statistics, validation_labels, classes)
        anchor_checkpoint = None
        embedding_source = f"encoder:{args.encoder}"

    metadata = {
        "key": key,
        "dataset": dataset,
        "scenario": scenario(setting),
        "n_players": n_players,
        "n_train": len(all_indices),
        "n_validation": len(validation_indices),
        "client_sizes": [len(groups[player]) for player in range(n_players)],
        "anchor_checkpoint": anchor_checkpoint,
        "embedding_source": embedding_source,
        "kernel_temperature": args.kernel_temperature,
    }
    return parzen, metadata
