"""Wall-clock stage timing shared by both methods and the demo.

`StageTimer` accumulates seconds per named stage; `to_dict()` emits the one
timing record shape used across `configs/`, `results/` and the demo cache:

    {"total_seconds": float, "stages": {name: seconds}, "rounds": int | None,
     "seeds": int | None, "source": str}
"""

from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter


class StageTimer:
    def __init__(self):
        self.stages: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str):
        started = perf_counter()
        try:
            yield
        finally:
            self.add(name, perf_counter() - started)

    def add(self, name: str, seconds: float) -> None:
        self.stages[name] = self.stages.get(name, 0.0) + float(seconds)

    @property
    def total_seconds(self) -> float:
        return float(sum(self.stages.values()))

    def to_dict(self, *, rounds: int | None = None, seeds: int | None = 1,
                source: str = "measured") -> dict:
        return {
            "total_seconds": self.total_seconds,
            "stages": dict(self.stages),
            "rounds": rounds,
            "seeds": seeds,
            "source": source,
        }
