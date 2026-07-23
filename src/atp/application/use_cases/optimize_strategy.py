"""Use case: optimize a strategy's parameters on one symbol."""

from __future__ import annotations

import asyncio
from datetime import timedelta

from atp.application.use_cases.load_price_history import LoadPriceHistory
from atp.domain.models.market import BarInterval
from atp.domain.models.optimization import (
    OptimizationResult,
    OptimizationSpec,
    WalkForwardResult,
)
from atp.domain.ports.optimization import StrategyOptimizer

_HISTORY_HEADROOM = 150  # want plenty of bars beyond the widest warmup

# Optimization needs more history than a single backtest: the search trains on
# a fraction of it and still has to fit the widest parameter's warmup.
DEFAULT_OPTIMIZATION_LOOKBACK = timedelta(days=5 * 365)


class OptimizeStrategy:
    def __init__(self, history: LoadPriceHistory, optimizer: StrategyOptimizer) -> None:
        self._history = history
        self._optimizer = optimizer

    async def execute(
        self,
        spec: OptimizationSpec,
        symbol: str,
        interval: BarInterval = BarInterval.DAY_1,
        *,
        n_trials: int = 50,
        holdout_fraction: float = 0.3,
        initial_cash: float = 10_000.0,
        commission: float = 0.001,
        seed: int = 42,
        lookback: timedelta | None = None,
    ) -> OptimizationResult:
        history = await self._history.execute(
            symbol,
            interval,
            min_bars=spec.strategy.min_history_bars + _HISTORY_HEADROOM,
            lookback=lookback or DEFAULT_OPTIMIZATION_LOOKBACK,
        )
        return await asyncio.to_thread(
            self._optimizer.optimize,
            spec,
            history,
            n_trials=n_trials,
            holdout_fraction=holdout_fraction,
            initial_cash=initial_cash,
            commission=commission,
            seed=seed,
        )

    async def walk_forward(
        self,
        spec: OptimizationSpec,
        symbol: str,
        interval: BarInterval = BarInterval.DAY_1,
        *,
        folds: int = 4,
        trials_per_fold: int = 25,
        initial_cash: float = 10_000.0,
        commission: float = 0.001,
        seed: int = 42,
        lookback: timedelta | None = None,
    ) -> WalkForwardResult:
        history = await self._history.execute(
            symbol,
            interval,
            min_bars=spec.strategy.min_history_bars + _HISTORY_HEADROOM,
            lookback=lookback or DEFAULT_OPTIMIZATION_LOOKBACK,
        )
        return await asyncio.to_thread(
            self._optimizer.walk_forward,
            spec,
            history,
            folds=folds,
            trials_per_fold=trials_per_fold,
            initial_cash=initial_cash,
            commission=commission,
            seed=seed,
        )
