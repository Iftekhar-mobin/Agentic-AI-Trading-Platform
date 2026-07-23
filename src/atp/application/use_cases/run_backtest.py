"""Use case: backtest a declarative strategy on one symbol."""

from __future__ import annotations

import asyncio
from datetime import timedelta

from atp.application.use_cases.load_price_history import LoadPriceHistory
from atp.domain.models.backtest import BacktestResult
from atp.domain.models.market import BarInterval
from atp.domain.models.strategy import StrategyDefinition
from atp.domain.ports.backtesting import BacktestEngine


class RunBacktest:
    def __init__(self, history: LoadPriceHistory, engine: BacktestEngine) -> None:
        self._history = history
        self._engine = engine

    async def execute(
        self,
        strategy: StrategyDefinition,
        symbol: str,
        interval: BarInterval = BarInterval.DAY_1,
        *,
        initial_cash: float = 10_000.0,
        commission: float = 0.001,
        lookback: timedelta | None = None,
    ) -> BacktestResult:
        history = await self._history.execute(
            symbol, interval, min_bars=strategy.min_history_bars, lookback=lookback
        )
        # The engine is CPU-bound; keep the event loop responsive.
        return await asyncio.to_thread(
            self._engine.run,
            strategy,
            history,
            initial_cash=initial_cash,
            commission=commission,
        )
