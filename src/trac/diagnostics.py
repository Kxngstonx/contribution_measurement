"""Evaluation helpers for contribution-scoring experiments.

Lives in ``src/`` (not a driver script) because ``scripts/run_tests.py`` only
puts ``PROJECT/src`` on ``sys.path`` and metrics that decide an experiment's
conclusion must be unit-tested.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np


def contaminated_from_overrides(
    client_indices: Sequence[Sequence[int]],
    overrides: Mapping[int | str, int],
) -> np.ndarray:
    """Boolean[n]: a client is contaminated if any of its indices is relabelled."""

    overridden = {int(index) for index in overrides}
    return np.asarray(
        [
            any(int(index) in overridden for index in indices)
            for indices in client_indices
        ],
        dtype=bool,
    )


def label_entropy(labels: Sequence[int]) -> float:
    """Shannon entropy (nats) of a label multiset."""

    values = np.asarray(labels)
    if values.size == 0:
        return 0.0
    _, counts = np.unique(values, return_counts=True)
    probabilities = counts / counts.sum()
    return float(-np.sum(probabilities * np.log(probabilities)))


def low_entropy_clients(
    client_labels: Sequence[Sequence[int]], k: int
) -> np.ndarray:
    """Indices of the k clients with the LOWEST label entropy (most skewed)."""

    entropies = np.asarray([label_entropy(labels) for labels in client_labels])
    return np.sort(np.argsort(entropies, kind="stable")[:k])


def contaminated_mask(
    client_indices: Sequence[Sequence[int]],
    overrides: Mapping[int | str, int],
    client_labels: Sequence[Sequence[int]] | None,
    *,
    expected_k: int,
) -> np.ndarray:
    """Boolean[n] contamination mask, dispatched by partition type.

    Mislabel settings carry a non-empty ``overrides`` -> override-derived.
    Non-IID settings have empty ``overrides`` -> the ``expected_k`` most
    label-skewed clients (by entropy). Asserts the group size matches
    ``expected_k`` so a wrong parse fails loudly.
    """

    if overrides:
        mask = contaminated_from_overrides(client_indices, overrides)
    else:
        if client_labels is None:
            raise ValueError("client_labels required when overrides is empty")
        n = len(client_indices)
        skewed = set(low_entropy_clients(client_labels, expected_k).tolist())
        mask = np.asarray([i in skewed for i in range(n)], dtype=bool)
    if int(mask.sum()) != expected_k:
        raise ValueError(
            f"contamination group size {int(mask.sum())} != expected {expected_k}"
        )
    return mask


def group_auc(scores: Sequence[float], contaminated: Sequence[bool]) -> float:
    """Mann-Whitney P(score_clean > score_contaminated); ties count 0.5.

    1.0 = perfect separation, 0.5 = chance, 0.0 = fully inverted. Exact
    (all pairs), which is fine for n <= 10.
    """

    scores = np.asarray(scores, dtype=np.float64)
    contaminated = np.asarray(contaminated, dtype=bool)
    clean_scores = scores[~contaminated]
    bad_scores = scores[contaminated]
    if clean_scores.size == 0 or bad_scores.size == 0:
        return float("nan")
    wins = 0.0
    for clean in clean_scores:
        wins += np.sum(clean > bad_scores) + 0.5 * np.sum(clean == bad_scores)
    return float(wins / (clean_scores.size * bad_scores.size))


def pcc_ceiling(
    ground_truth: Sequence[float], contaminated: Sequence[bool]
) -> float:
    """PCC a perfect group separator scores: pcc(1 - contaminated, ground_truth)."""

    ground_truth = np.asarray(ground_truth, dtype=np.float64)
    indicator = 1.0 - np.asarray(contaminated, dtype=np.float64)
    if ground_truth.std() < 1e-12 or indicator.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(indicator, ground_truth)[0, 1])


def flip_round(pcc_series: Sequence[float]) -> int | None:
    """First index t where the series goes < 0 and stays < 0 through the end."""

    values = list(pcc_series)
    for start in range(len(values)):
        value = values[start]
        if not math.isfinite(value) or value >= 0:
            continue
        if all((not math.isfinite(v)) or v < 0 for v in values[start:]):
            return start
    return None
