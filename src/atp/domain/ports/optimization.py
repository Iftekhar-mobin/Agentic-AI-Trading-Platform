"""Port for strategy optimizers."""

from __future__ import annotations

from typing import Protocol

from atp.domain.models.market import PriceHistory
from atp.domain.models.optimization import (
    OptimizationResult,
    OptimizationSpec,
    WalkForwardResult,
)


class StrategyOptimizer(Protocol):
    """Searches a declarative parameter space for a strategy.

    Synchronous by design (CPU-bound); callers offload to a worker thread.
    ``optimize`` must hold out the final ``holdout_fraction`` of history from
    the search and report out-of-sample performance on it. ``walk_forward``
    re-optimizes per fold and reports only out-of-sample fold results.
    """

    def optimize(
        self,
        spec: OptimizationSpec,
        history: PriceHistory,
        *,
        n_trials: int = 50,
        holdout_fraction: float = 0.3,
        initial_cash: float = 10_000.0,
        commission: float = 0.001,
        seed: int = 42,
    ) -> OptimizationResult: ...

    def walk_forward(
        self,
        spec: OptimizationSpec,
        history: PriceHistory,
        *,
        folds: int = 4,
        trials_per_fold: int = 25,
        initial_cash: float = 10_000.0,
        commission: float = 0.001,
        seed: int = 42,
    ) -> WalkForwardResult: ...
