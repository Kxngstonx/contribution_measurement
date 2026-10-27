"""Small helpers for materialized coalition games and evaluation."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np


def players_from_mask(mask: int, n_players: int) -> frozenset[int]:
    return frozenset(
        player
        for player in range(n_players)
        if mask & (1 << player)
    )


def mask_from_players(players) -> int:
    mask = 0
    for player in players:
        mask |= 1 << int(player)
    return mask


def table_utility(values: Sequence[float]):
    table = np.asarray(values, dtype=np.float64)
    size = len(table)
    if size < 4 or size & (size - 1):
        raise ValueError("coalition table length must be a power of two")

    def utility(players: frozenset[int]) -> float:
        return float(table[mask_from_players(players)])

    return utility


def pcc(left, right) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("PCC vectors must have identical shapes")
    if left.std() < 1e-12 or right.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def rmse(left, right) -> float:
    error = np.asarray(left, dtype=np.float64) - np.asarray(
        right, dtype=np.float64
    )
    return float(np.sqrt(np.mean(np.square(error))))


def stable_seed(key: str, seed: int) -> int:
    digest = hashlib.sha256(key.encode()).digest()
    return int(seed) + int.from_bytes(digest[:4], "big")
