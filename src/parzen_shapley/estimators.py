"""Target-free Monte Carlo estimators for coalition Shapley values."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from time import perf_counter

import numpy as np


Utility = Callable[[frozenset[int]], float]


class CountingUtility:
    """Cache utility calls and count distinct nonempty coalition queries."""

    def __init__(self, utility: Utility):
        self.utility = utility
        self.cache: dict[frozenset[int], float] = {}
        self.nonempty_queries: set[tuple[int, ...]] = set()

    def __call__(self, players: Iterable[int]) -> float:
        coalition = frozenset(int(player) for player in players)
        if coalition not in self.cache:
            self.cache[coalition] = float(self.utility(coalition))
        if coalition:
            self.nonempty_queries.add(tuple(sorted(coalition)))
        return self.cache[coalition]


def exact_shapley(utility: Utility, n_players: int) -> np.ndarray:
    """Compute exact Shapley in O(n 2^n) utility accesses."""

    result = np.zeros(n_players, dtype=np.float64)
    denominator = math.factorial(n_players)
    for player in range(n_players):
        bit = 1 << player
        for mask in range(1 << n_players):
            if mask & bit:
                continue
            size = mask.bit_count()
            weight = (
                math.factorial(size)
                * math.factorial(n_players - size - 1)
                / denominator
            )
            before = frozenset(
                index
                for index in range(n_players)
                if mask & (1 << index)
            )
            result[player] += weight * (
                utility(before | {player}) - utility(before)
            )
    return result


def raw_marginals(
    permutation: np.ndarray, utility: CountingUtility
) -> np.ndarray:
    output = np.zeros(len(permutation), dtype=np.float64)
    prefix = frozenset()
    before = utility(prefix)
    for player in permutation:
        after_set = prefix | {int(player)}
        after = utility(after_set)
        output[player] = after - before
        prefix = after_set
        before = after
    return output


def two_additive_control(
    utility: CountingUtility, n_players: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    empty = utility(frozenset())
    singleton = np.asarray(
        [
            utility(frozenset({player})) - empty
            for player in range(n_players)
        ],
        dtype=np.float64,
    )
    pair = np.zeros((n_players, n_players), dtype=np.float64)
    for left in range(n_players):
        for right in range(left + 1, n_players):
            interaction = (
                utility(frozenset({left, right}))
                - empty
                - singleton[left]
                - singleton[right]
            )
            pair[left, right] = interaction
            pair[right, left] = interaction
    control_phi = singleton + 0.5 * pair.sum(axis=1)
    return singleton, pair, control_phi


def residual_marginals(
    permutation: np.ndarray,
    utility: CountingUtility,
    singleton: np.ndarray,
    pair: np.ndarray,
) -> np.ndarray:
    output = np.zeros(len(permutation), dtype=np.float64)
    prefix = frozenset()
    before = utility(prefix)
    for player in permutation:
        after_set = prefix | {int(player)}
        after = utility(after_set)
        control_marginal = singleton[player] + sum(
            pair[player, previous] for previous in prefix
        )
        output[player] = after - before - control_marginal
        prefix = after_set
        before = after
    return output


def balanced_permutations(base: np.ndarray):
    """Yield every cyclic rotation of a base order and its reverse."""

    base = np.asarray(base, dtype=np.int64)
    for direction in (base, base[::-1]):
        for shift in range(len(base)):
            yield np.roll(direction, -shift)


def position_counts(base: np.ndarray) -> np.ndarray:
    counts = np.zeros((len(base), len(base)), dtype=np.int64)
    for permutation in balanced_permutations(base):
        for position, player in enumerate(permutation):
            counts[player, position] += 1
    return counts


def _finalize(
    counted: CountingUtility,
    prediction: np.ndarray,
    *,
    n_players: int,
    paths: int,
    method: str,
    started: float | None = None,
) -> dict:
    expected = (1 << n_players) - 1
    exhaustive = len(counted.nonempty_queries) == expected
    if exhaustive:
        prediction = exact_shapley(counted, n_players)
    empty = counted(frozenset())
    full = counted(frozenset(range(n_players)))
    total = full - empty
    prediction = np.asarray(prediction, dtype=np.float64)
    prediction += (total - float(prediction.sum())) / n_players
    return {
        "method": method,
        "phi": prediction,
        "paths": int(paths),
        "unique_nonempty_queries": len(counted.nonempty_queries),
        "expected_nonempty_coalitions": expected,
        "exhaustive_cache": exhaustive,
        "efficiency_error": float(prediction.sum() - total),
        "seconds": (perf_counter() - started) if started is not None else None,
    }


def vanilla_mc(
    utility: Utility,
    n_players: int,
    *,
    paths: int,
    seed: int,
) -> dict:
    """Independent uniform permutation Monte Carlo."""

    started = perf_counter()
    if paths < 1:
        raise ValueError("paths must be positive")
    counted = CountingUtility(utility)
    generator = np.random.default_rng(seed)
    samples = [
        raw_marginals(generator.permutation(n_players), counted)
        for _ in range(paths)
    ]
    return _finalize(
        counted,
        np.mean(samples, axis=0),
        n_players=n_players,
        paths=paths,
        method="vanilla_mc",
        started=started,
    )


def antithetic_mc(
    utility: Utility,
    n_players: int,
    *,
    paths: int,
    seed: int,
) -> dict:
    """Uniform permutations paired with their reverse."""

    started = perf_counter()
    if paths < 2 or paths % 2:
        raise ValueError("antithetic paths must be a positive even number")
    counted = CountingUtility(utility)
    generator = np.random.default_rng(seed)
    samples = []
    for _ in range(paths // 2):
        permutation = generator.permutation(n_players)
        samples.append(raw_marginals(permutation, counted))
        samples.append(raw_marginals(permutation[::-1], counted))
    return _finalize(
        counted,
        np.mean(samples, axis=0),
        n_players=n_players,
        paths=paths,
        method="antithetic_mc",
        started=started,
    )


def control_mc(
    utility: Utility,
    n_players: int,
    *,
    paths: int,
    seed: int,
) -> dict:
    """Independent permutation MC on a two-additive residual game."""

    started = perf_counter()
    if paths < 1:
        raise ValueError("paths must be positive")
    counted = CountingUtility(utility)
    singleton, pair, control_phi = two_additive_control(
        counted, n_players
    )
    generator = np.random.default_rng(seed)
    residuals = [
        residual_marginals(
            generator.permutation(n_players),
            counted,
            singleton,
            pair,
        )
        for _ in range(paths)
    ]
    return _finalize(
        counted,
        control_phi + np.mean(residuals, axis=0),
        n_players=n_players,
        paths=paths,
        method="control_mc",
        started=started,
    )


def balanced_mc(
    utility: Utility,
    n_players: int,
    *,
    paths: int,
    seed: int,
    control: bool,
) -> dict:
    """Cyclic/reverse position-balanced Monte Carlo.

    ``paths`` must be an integer number of blocks, where each block contains
    all ``2 * n_players`` rotations of one uniform base permutation.
    """

    started = perf_counter()
    block_size = 2 * n_players
    if paths < block_size or paths % block_size:
        raise ValueError(
            f"balanced paths must be a positive multiple of {block_size}"
        )
    counted = CountingUtility(utility)
    if control:
        singleton, pair, control_phi = two_additive_control(
            counted, n_players
        )
    generator = np.random.default_rng(seed)
    blocks = []
    for _ in range(paths // block_size):
        base = generator.permutation(n_players)
        if not np.all(position_counts(base) == 2):
            raise AssertionError("balanced block has invalid position counts")
        if control:
            samples = [
                residual_marginals(
                    permutation, counted, singleton, pair
                )
                for permutation in balanced_permutations(base)
            ]
        else:
            samples = [
                raw_marginals(permutation, counted)
                for permutation in balanced_permutations(base)
            ]
        blocks.append(np.mean(samples, axis=0))
    estimate = np.mean(blocks, axis=0)
    if control:
        estimate = control_phi + estimate
    return _finalize(
        counted,
        estimate,
        n_players=n_players,
        paths=paths,
        method="balanced_control_mc" if control else "balanced_raw_mc",
        started=started,
    )
