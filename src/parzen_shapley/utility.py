"""Class-conditional Parzen coalition utility on frozen embeddings.

The utility is split into two halves along the FL trust boundary:

- `client_local_statistics` is the only function a client-local process may
  call.  It never takes `validation_labels` as an argument -- that is what
  structurally keeps validation-set labels out of client hands -- and it
  only ever sees *this* client's own rows.  Its return value
  (`ClientStatistics`) is an aggregated per-class kernel-density sum, never a
  raw or per-example embedding.
- `assemble_parzen_utility` is the only function that needs
  `validation_labels`, and is meant to be called by whichever party holds
  the labeled probe set (the "server").  It reconstructs the coalition-query
  closure purely by summing the per-client statistics it receives, relying
  on the fact that the utility is exactly additive over clients.

`make_parzen_utility` keeps its original signature/behaviour unchanged: it is
now a thin single-process simulation that normalizes the validation set once
("server" broadcast prep), calls `client_local_statistics` once per client on
that client's own row slice, and finishes with `assemble_parzen_utility`.
This keeps it byte-for-byte equivalent to the previous monolithic
implementation (see `tests/test_utility.py`, unmodified, and
`tests/test_utility_split.py`).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import NamedTuple

import torch
import torch.nn.functional as F


class ClientStatistics(NamedTuple):
    """One client's aggregated contribution to the coalition utility.

    `density_sum[v, c]` is `K_{i,c}(z_v) = sum_{x in D_{i,c}} k_tau(z_v, x)`
    and `class_count[c]` is `|D_{i,c}|`, this client's own class-`c` sample
    count. Both are already summed/aggregated -- no raw or per-example
    embedding survives past this point.
    """

    density_sum: torch.Tensor  # (n_validation, n_classes)
    class_count: torch.Tensor  # (n_classes,)


def client_local_statistics(
    train_embeddings: torch.Tensor,
    train_labels: torch.Tensor,
    validation_embeddings: torch.Tensor,
    classes: torch.Tensor,
    *,
    kernel_temperature: float = 20.0,
    batch_size: int = 512,
    device: str | torch.device | None = None,
) -> ClientStatistics:
    """Compute one client's per-class kernel-density sums.

    `train_embeddings`/`train_labels` must be this client's own rows only.
    `validation_embeddings` is the broadcast probe set, already L2-normalized
    by the caller (the server) -- this function does not need, and does not
    accept, the associated `validation_labels`.
    """

    if kernel_temperature <= 0.0:
        raise ValueError("kernel_temperature must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    dev = torch.device(
        device if device is not None else train_embeddings.device
    )
    x = F.normalize(train_embeddings.detach().float(), p=2, dim=-1).to(dev)
    y = train_labels.detach().reshape(-1).to(dev)
    if x.shape[0] == 0:
        raise ValueError("every client must contain at least one row")
    validation = validation_embeddings.to(dev)
    classes = classes.to(dev)

    counts = torch.stack([(y == label).sum() for label in classes])
    batches = []
    for start in range(0, len(validation), batch_size):
        similarity = validation[start : start + batch_size] @ x.t()
        kernels = torch.exp(
            float(kernel_temperature) * (similarity - 1.0)
        )
        batches.append(
            torch.stack(
                [
                    kernels[:, y == label].sum(dim=1)
                    if bool((y == label).any())
                    else torch.zeros(
                        len(similarity),
                        device=dev,
                        dtype=kernels.dtype,
                    )
                    for label in classes
                ],
                dim=1,
            )
        )
    density_sum = torch.cat(batches, dim=0)
    return ClientStatistics(density_sum=density_sum, class_count=counts)


def assemble_parzen_utility(
    client_statistics: Sequence[ClientStatistics],
    validation_labels: torch.Tensor,
    classes: torch.Tensor,
) -> Callable[[frozenset[int]], float]:
    """Reconstruct the coalition-utility closure from per-client statistics.

    Only the party holding `validation_labels` in plaintext should call
    this -- clients themselves never see it (see `client_local_statistics`).
    """

    if len(client_statistics) == 0:
        raise ValueError("client_statistics must not be empty")
    dev = client_statistics[0].density_sum.device
    validation_labels = validation_labels.detach().reshape(-1).to(dev)
    classes = classes.to(dev)

    density_by_client = torch.stack(
        [statistics.density_sum.to(dev) for statistics in client_statistics]
    )
    counts_by_client = torch.stack(
        [statistics.class_count.to(dev) for statistics in client_statistics]
    ).to(density_by_client.dtype)
    true_columns = (
        validation_labels.unsqueeze(1) == classes.unsqueeze(0)
    ).to(density_by_client.dtype)
    cache: dict[frozenset[int], float] = {frozenset(): 0.0}

    def utility(players: frozenset[int]) -> float:
        coalition = frozenset(int(player) for player in players)
        if coalition in cache:
            return cache[coalition]
        selected = torch.as_tensor(
            sorted(coalition), dtype=torch.long, device=dev
        )
        density_sum = density_by_client[selected].sum(dim=0)
        class_count = counts_by_client[selected].sum(dim=0)
        mean_density = density_sum / class_count.clamp_min(1.0).unsqueeze(0)
        logits = torch.log(mean_density.clamp_min(1e-30))
        posterior = torch.softmax(logits, dim=1)
        score = float((posterior * true_columns).sum(dim=1).mean())
        cache[coalition] = score
        return score

    utility.cache = cache  # type: ignore[attr-defined]
    utility.density_by_client = density_by_client  # type: ignore[attr-defined]
    utility.counts_by_client = counts_by_client  # type: ignore[attr-defined]
    return utility


def make_parzen_utility(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    client_rows: Sequence[Sequence[int]],
    validation_embeddings: torch.Tensor,
    validation_labels: torch.Tensor,
    *,
    kernel_temperature: float = 20.0,
    batch_size: int = 512,
    device: str | torch.device | None = None,
) -> Callable[[frozenset[int]], float]:
    """Build a cached Parzen utility over client coalitions.

    Single-process simulation of the client-local/server-assemble split:
    the validation set is normalized once (server broadcast prep), then each
    client's own row slice is passed through `client_local_statistics`
    independently, and `assemble_parzen_utility` reconstructs the same
    coalition-query closure the previous monolithic implementation returned.
    """

    if kernel_temperature <= 0.0:
        raise ValueError("kernel_temperature must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    dev = torch.device(device if device is not None else embeddings.device)
    train_labels = labels.detach().reshape(-1).to(dev)
    validation = F.normalize(
        validation_embeddings.detach().float(), p=2, dim=-1
    ).to(dev)
    validation_labels = validation_labels.detach().reshape(-1).to(dev)
    classes = torch.unique(validation_labels, sorted=True)
    if classes.numel() == 0:
        raise ValueError("validation_labels must not be empty")

    train_raw = embeddings.detach()
    statistics = []
    for rows in client_rows:
        row_ids_raw = torch.as_tensor(
            sorted(int(row) for row in rows),
            dtype=torch.long,
            device=train_raw.device,
        )
        if row_ids_raw.numel() == 0:
            raise ValueError("every client must contain at least one row")
        row_ids_dev = row_ids_raw.to(dev)
        statistics.append(
            client_local_statistics(
                train_raw[row_ids_raw],
                train_labels[row_ids_dev],
                validation,
                classes,
                kernel_temperature=kernel_temperature,
                batch_size=batch_size,
                device=dev,
            )
        )
    return assemble_parzen_utility(statistics, validation_labels, classes)
