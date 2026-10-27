"""Paths, defaults and environment probes for the demo."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT / "src"))

# The scripts default to ../SPACE/data; the demo always uses the repo's own copy.
DATA_ROOT = PROJECT / "data"
REFERENCE_DIR = PROJECT / "assets" / "reference"
ANCHOR_DIR = PROJECT / "artifacts" / "anchors"
CACHE_DIR = PROJECT / "artifacts" / "demo_cache"
MANIFEST = PROJECT / "configs" / "space_exact_gt_manifest.json"
GT_JSON = PROJECT / "results" / "ground_truth_group_shapley_sweep.json"
# Stored TRAC runs at the current hyperparameters (batch 128): mnist/fmnist from
# the batch-128 rerun, cifar from the 24-setting run.
TRAC_JSON = {
    "mnist": PROJECT / "results" / "fltrust_structure_b128.json",
    "fmnist": PROJECT / "results" / "fltrust_structure_b128.json",
    "cifar": PROJECT / "results" / "fltrust_structure_cw24.json",
}
TRAC_TRAJ_DIR = {
    "mnist": PROJECT / "results" / "fltrust_structure_b128_trajectories",
    "fmnist": PROJECT / "results" / "fltrust_structure_b128_trajectories",
    "cifar": PROJECT / "results" / "fltrust_structure_cw24_trajectories",
}

DATASETS = ("mnist", "fmnist", "cifar")
DATASET_TITLES = {"mnist": "MNIST", "fmnist": "Fashion-MNIST", "cifar": "CIFAR-10"}
DEFAULT_CHOICE = {"mnist": "mislabel-4", "fmnist": "noniid-4", "cifar": "mislabel-2"}

DEFAULT_TRAC_ROUNDS = 5
MAX_TRAC_ROUNDS = 40
# TRAC runs 30 rounds on every dataset. The stored CIFAR-10 run (cw24) went to 40;
# replay truncates its trajectory to round 30.
FULL_TRAC_ROUNDS = {"mnist": 30, "fmnist": 30, "cifar": 30}
# Measured on an RTX 3080 at batch 128 (results/fltrust_structure_b128.json, cw24.json).
SECONDS_PER_ROUND = {"mnist": 18.6, "fmnist": 18.6, "cifar": 7.7}
REPLAY_ROUND_DELAY = 0.15  # seconds per animated round in replay mode
REPLAY_STAGE_DELAY = 0.3   # seconds per Parzen stage in replay mode

# Tk in this environment has no Xft, so Korean is rendered through PIL with a
# TrueType font. Pretendard (SIL OFL, demo/fonts/) ships with the demo; the
# rest are fallbacks (Windows fonts are readable from WSL).
FONT_DIR = PROJECT / "demo" / "fonts"
FONT_CANDIDATES = [
    str(FONT_DIR / "Pretendard-Regular.ttf"),
    "/mnt/c/Windows/Fonts/malgun.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]
FONT_BOLD_CANDIDATES = [
    str(FONT_DIR / "Pretendard-Bold.ttf"),
    "/mnt/c/Windows/Fonts/malgunbd.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
]


def korean_font_path(bold: bool = False) -> str | None:
    candidates = (FONT_BOLD_CANDIDATES + FONT_CANDIDATES) if bold else FONT_CANDIDATES
    for candidate in candidates:
        if Path(candidate).is_file():
            return candidate
    return None


def detect_device() -> str:
    """'cuda' when a GPU is usable, else 'cpu'. Imports torch lazily."""

    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def device_label(device: str) -> str:
    if device == "cuda":
        import torch

        return f"GPU: {torch.cuda.get_device_name(0)}"
    return "CPU"


def estimated_trac_seconds(rounds: int) -> float:
    return rounds * sum(SECONDS_PER_ROUND.values())


def full_trac_seconds() -> float:
    return sum(FULL_TRAC_ROUNDS[d] * SECONDS_PER_ROUND[d] for d in DATASETS)
