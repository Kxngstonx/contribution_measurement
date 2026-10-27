"""One TRAC communication round (needs model/Dataset).

TRAC (Target-distribution Reweighted server Alignment for Contribution
measurement) keeps a SINGLE global model. Each round, from the shared global
state w_t:

1. every client i trains E local epochs and sends Delta_i = w_i - w_t
   (`local_pseudo_gradient`);
2. the server takes ONE full-batch SGD step on its root D_0 with the
   class-weighted loss (`scoring.class_balanced_weights`: every class in the
   root counts equally, i.e. a uniform target class distribution) and obtains
   Delta_0 = -lr * grad L_0(w_t) (`root_pseudo_gradient`);
3. c_i = cos(Delta_i, Delta_0) (`scoring.root_cosines`; raw, may be negative);
4. s_i <- alpha * s_i + (1 - alpha) * c_i (`scoring.cosine_ema`; round 0 sets
   s_i = c_i). s_i is the contribution score;
5. a_i = s_i / sum_j |s_j| (`scoring.signed_weights`): the score is also the
   aggregation weight, so a client whose s_i is negative is subtracted;
6. w_{t+1} = w_t + sum_i a_i * gamma * Delta_i / ||Delta_i||
   (`normalize_pseudo_gradient` to the fixed `gamma_target`,
   `weighted_aggregate`, `apply_update`). There is no outer learning rate:
   gamma fixes the global step, independent of how much the server trained.

Because the aggregation weight is the running score, the global trajectory
depends on past scores. The root direction Delta_0 itself is recomputed from
the server data alone every round and is never reweighted by scores.

`evaluate_accuracy` scores a global state on ground-truth labels.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from parzen_shapley.anchors import IndexedRelabelledDataset, cpu_state_dict, train_anchor
from parzen_shapley.games import stable_seed

from .scoring import (
    apply_update,
    cosine_ema,
    cosine_similarity,
    normalize_pseudo_gradient,
    pseudo_gradient_norm,
    root_cosines,
    signed_weights,
    weighted_aggregate,
)


def local_pseudo_gradient(
    base_state: Mapping[str, Tensor],
    model_factory: Callable[[], nn.Module],
    dataset: Dataset,
    indices: Sequence[int],
    overrides: Mapping[int | str, int],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str | torch.device,
    loader_seed: int,
    class_weights: Sequence[float] | None = None,
) -> dict[str, Tensor]:
    """Delta_i = (state after E local epochs) - (state before), on CPU.

    Reuses `anchors.train_anchor` (called with `momentum=0.0`) as the local
    training step -- an E-epoch local update, not a single gradient step.
    `class_weights` turns the loss into a per-class weighted mean (used by
    the root only; clients never pass it).
    """

    model = model_factory()
    model.load_state_dict({key: tensor.clone() for key, tensor in base_state.items()})
    trained = train_anchor(
        model,
        dataset,
        indices,
        overrides,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        momentum=0.0,
        device=device,
        loader_seed=loader_seed,
        class_weights=class_weights,
    )
    after = cpu_state_dict(trained)
    before = {key: tensor.detach().cpu() for key, tensor in base_state.items()}
    return {name: after[name] - before[name] for name in before}


def root_pseudo_gradient(
    base_state: Mapping[str, Tensor],
    model_factory: Callable[[], nn.Module],
    root_dataset: Dataset,
    root_indices: Sequence[int],
    *,
    learning_rate: float,
    device: str | torch.device,
    loader_seed: int,
    class_weights: Sequence[float] | None = None,
) -> dict[str, Tensor]:
    """Delta_0 = -learning_rate * grad L_0(base_state): ONE full-batch SGD step
    on the root, on CPU.

    `train_anchor`'s DataLoader has no `drop_last`, so `batch_size =
    len(root_indices)` yields exactly one batch; `shuffle=True` only permutes
    rows inside it and the (weighted) mean loss is permutation-invariant, so
    the direction is independent of `loader_seed` (float32 summation order
    aside). `class_weights` reweights the loss per class
    (`scoring.class_balanced_weights`); None keeps the plain mean.
    """

    return local_pseudo_gradient(
        base_state,
        model_factory,
        root_dataset,
        root_indices,
        {},
        epochs=1,
        batch_size=len(root_indices),
        learning_rate=learning_rate,
        device=device,
        loader_seed=loader_seed,
        class_weights=class_weights,
    )


@dataclass
class RoundOutput:
    global_state: dict[str, Tensor]
    a: np.ndarray
    raw_cos: np.ndarray
    cos_ema: np.ndarray
    server_norm: float
    aggregate_norm: float
    anchor_vs_aggregate_cos: float
    client_norm_mean: float
    client_norms: np.ndarray
    n_negative: int
    seconds: float = 0.0  # wall clock of this round (client + root steps, scoring, aggregation)


def run_round(
    global_state: Mapping[str, Tensor],
    cos_ema_prev: np.ndarray,
    iteration: int,
    *,
    model_factory: Callable[[], nn.Module],
    train_dataset: Dataset,
    server_dataset: Dataset,
    client_indices: Sequence[Sequence[int]],
    server_indices: Sequence[int],
    overrides: Mapping[int | str, int],
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    gamma_target: float,
    alpha: float,
    server_class_weights: Sequence[float] | None = None,
    device: str | torch.device,
    key: str,
    master_seed: int,
) -> RoundOutput:
    """One TRAC round; see the module docstring for the six steps.

    `cos_ema_prev` is ignored at `iteration == 0` (s_i starts at c_i).
    `server_class_weights=None` is the unweighted-root-loss ablation; client
    training is never reweighted. `n_negative` counts clients whose score
    s_i is negative this round (their updates were subtracted).
    """

    started = perf_counter()
    n_players = len(client_indices)

    deltas = []
    for i in range(n_players):
        loader_seed = stable_seed(
            f"{key}#fltrust_structure#r{iteration}#c{i}", master_seed
        )
        deltas.append(
            local_pseudo_gradient(
                global_state,
                model_factory,
                train_dataset,
                client_indices[i],
                overrides,
                epochs=local_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                device=device,
                loader_seed=loader_seed,
            )
        )

    server_seed = stable_seed(
        f"{key}#fltrust_structure#r{iteration}#server", master_seed
    )
    server_delta = root_pseudo_gradient(
        global_state,
        model_factory,
        server_dataset,
        server_indices,
        learning_rate=learning_rate,
        device=device,
        loader_seed=server_seed,
        class_weights=server_class_weights,
    )
    server_norm = pseudo_gradient_norm(server_delta)

    raw_cos = root_cosines(deltas, server_delta)
    if iteration == 0:
        cos_ema_next = raw_cos.copy()
    else:
        cos_ema_next = cosine_ema(cos_ema_prev, raw_cos, alpha=alpha)
    a = signed_weights(cos_ema_next)
    n_negative = int((cos_ema_next < 0.0).sum())

    client_norms = np.asarray(
        [pseudo_gradient_norm(delta) for delta in deltas], dtype=np.float64
    )
    normalized = [normalize_pseudo_gradient(delta, gamma_target) for delta in deltas]
    aggregate = weighted_aggregate(normalized, a)
    next_global = apply_update(global_state, aggregate)

    return RoundOutput(
        global_state=next_global,
        a=a,
        raw_cos=raw_cos,
        cos_ema=cos_ema_next,
        server_norm=server_norm,
        aggregate_norm=pseudo_gradient_norm(aggregate),
        anchor_vs_aggregate_cos=cosine_similarity(server_delta, aggregate),
        client_norm_mean=float(client_norms.mean()),
        client_norms=client_norms,
        n_negative=n_negative,
        seconds=perf_counter() - started,
    )


def evaluate_accuracy(
    state: Mapping[str, Tensor],
    model_factory: Callable[[], nn.Module],
    dataset: Dataset,
    indices: Sequence[int],
    *,
    batch_size: int,
    device: str | torch.device,
) -> float:
    """Top-1 accuracy of `state` on dataset[indices] with GROUND-TRUTH labels.

    Wraps in `IndexedRelabelledDataset(dataset, indices, {})` so mislabel
    overrides can never leak into the evaluation set. The FL client nets emit
    `log_softmax`, so `argmax` of the output equals `argmax` of the
    probabilities -- no softmax is applied here.
    """

    model = model_factory()
    model.load_state_dict({key: tensor.clone() for key, tensor in state.items()})
    target_device = torch.device(device)
    model.to(target_device)
    model.eval()

    loader = DataLoader(
        IndexedRelabelledDataset(dataset, indices, {}),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(target_device)
            predictions = model(images).argmax(dim=1).cpu()
            correct += int((predictions == labels).sum())
            total += int(labels.shape[0])
    if total == 0:
        raise ValueError("indices must not be empty")
    return correct / total
