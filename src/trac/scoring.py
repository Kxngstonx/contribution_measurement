"""Pseudo-gradient math for TRAC (`trac.scoring`).

Pure functions over state-dict-shaped pseudo-gradients: CPU only, no model
or dataset dependencies. The alignment signal is `cos(Delta_i, Delta_0)`
against a *server root* pseudo-gradient.

Sections:
- generic primitives (cosine, norm, fixed-target rescaling, weighted
  aggregation, state update);
- TRAC: `root_cosines` (c_i), `cosine_ema` (s_i, the contribution score),
  `signed_weights` (a_i = s_i / sum|s_j|, the aggregation weight) and
  `class_balanced_weights` (the target-distribution loss reweighting).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

_EPS = 1e-12

# Denominator floor for the normalized aggregation weights (FLTrust's
# `cos_sim / (sum(cos_sim) + 1e-9)` convention).
_TRUST_EPS = 1e-9


# --------------------------------------------------------------------------
# Pseudo-gradient primitives
# --------------------------------------------------------------------------


def cosine_similarity(
    a: Mapping[str, Tensor], b: Mapping[str, Tensor], eps: float = _EPS
) -> float:
    """Cosine similarity between two state-dict-shaped gradients."""

    if a.keys() != b.keys():
        raise ValueError("gradient dicts must share identical keys")
    dot = 0.0
    norm_a_sq = 0.0
    norm_b_sq = 0.0
    for key in a:
        left = a[key].reshape(-1).double()
        right = b[key].reshape(-1).double()
        dot += torch.dot(left, right).item()
        norm_a_sq += torch.dot(left, left).item()
        norm_b_sq += torch.dot(right, right).item()
    norm_a = math.sqrt(norm_a_sq)
    norm_b = math.sqrt(norm_b_sq)
    if norm_a < eps or norm_b < eps:
        return 0.0
    return dot / (norm_a * norm_b)


def pseudo_gradient_norm(delta: Mapping[str, Tensor]) -> float:
    """L2 norm of a state-dict-shaped pseudo-gradient, across all keys."""

    total = 0.0
    for tensor in delta.values():
        flattened = tensor.reshape(-1).double()
        total += torch.dot(flattened, flattened).item()
    return math.sqrt(total)


def normalize_pseudo_gradient(
    delta: Mapping[str, Tensor], gamma: float, eps: float = _EPS
) -> dict[str, Tensor]:
    """u_i = Gamma * delta / ||delta||; an all-zero delta stays all-zero."""

    norm = pseudo_gradient_norm(delta)
    if norm < eps:
        return {key: torch.zeros_like(tensor) for key, tensor in delta.items()}
    scale = gamma / norm
    return {key: tensor * scale for key, tensor in delta.items()}


def weighted_aggregate(
    client_updates: Sequence[Mapping[str, Tensor]], weights: Sequence[float]
) -> dict[str, Tensor]:
    """u_N = sum_i weights[i] * client_updates[i]."""

    if not client_updates:
        raise ValueError("client_updates must not be empty")
    if len(client_updates) != len(weights):
        raise ValueError("client_updates and weights must have equal length")
    keys = client_updates[0].keys()
    aggregate = {key: torch.zeros_like(client_updates[0][key]) for key in keys}
    for update, weight in zip(client_updates, weights):
        if update.keys() != keys:
            raise ValueError("all client updates must share identical keys")
        for key in keys:
            aggregate[key] = aggregate[key] + float(weight) * update[key]
    return aggregate


def apply_update(
    state: Mapping[str, Tensor], update: Mapping[str, Tensor]
) -> dict[str, Tensor]:
    """w_new = state + update, key-by-key. Raises on key-set mismatch."""

    if state.keys() != update.keys():
        raise ValueError("state and update must share identical keys")
    return {key: state[key] + update[key] for key in state}


# --------------------------------------------------------------------------
# TRAC: alignment score, EMA, signed aggregation weights, root loss weights
# --------------------------------------------------------------------------


def root_cosines(
    client_deltas: Sequence[Mapping[str, Tensor]],
    server_delta: Mapping[str, Tensor],
) -> np.ndarray:
    """c_i = cos(client_deltas[i], server_delta) -- raw, may be negative.

    `cosine_similarity` returns 0.0 when either operand has norm < 1e-12, so a
    converged/all-zero server_delta yields c == 0 for every client.
    """

    return np.asarray(
        [cosine_similarity(delta, server_delta) for delta in client_deltas],
        dtype=np.float64,
    )


def cosine_ema(prev: np.ndarray, raw_cos: np.ndarray, *, alpha: float) -> np.ndarray:
    """s_i^t = alpha * s_i^{t-1} + (1 - alpha) * c_i^t: plain EMA of the raw cosines.

    No clamp and no renormalization -- negative values survive. This is the
    contribution score: a single round's cosine is noisy, and a clamp would
    destroy the resolution among clients pinned to the floor.
    """

    prev = np.asarray(prev, dtype=np.float64)
    raw_cos = np.asarray(raw_cos, dtype=np.float64)
    return alpha * prev + (1.0 - alpha) * raw_cos


def signed_weights(scores: np.ndarray, *, eps: float = _TRUST_EPS) -> np.ndarray:
    """a_i = scores[i] / (sum_j |scores[j]| + eps): TRAC's aggregation weights.

    Negative scores keep their sign, so those clients' updates are SUBTRACTED.
    Dividing by the L1 norm (not the signed sum) keeps the global step <= gamma
    and never flips every weight's sign when the scores sum to <= 0. All zeros
    (no update) when every score is 0.
    """

    scores = np.asarray(scores, dtype=np.float64)
    weights = scores / (np.abs(scores).sum() + eps)
    assert np.isfinite(weights).all(), "signed weights must be finite"
    return weights


def class_balanced_weights(labels: Sequence[int], num_classes: int) -> np.ndarray:
    """Per-class loss weights `n / (M_present * n_c)` that make every present
    class count equally in a weighted-mean loss (a uniform target class
    distribution); absent classes get 0.

    Scaled so `sum_c n_c * w_c = n` (mean weight per sample 1). The scale is
    irrelevant to `F.nll_loss(weight=...)`'s weighted mean, which divides by
    the batch's summed weights.
    """

    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes)
    if counts.size != num_classes:
        raise ValueError(f"labels must lie in [0, {num_classes}), got max {counts.size - 1}")
    present = counts > 0
    if not present.any():
        raise ValueError("labels must not be empty")
    weights = np.zeros(num_classes, dtype=np.float64)
    weights[present] = counts.sum() / (present.sum() * counts[present])
    return weights
