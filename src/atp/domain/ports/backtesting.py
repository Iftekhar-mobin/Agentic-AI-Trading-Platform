"""Port for backtest engines."""

from __future__ import annotations

from typing import Protocol

from atp.domain.models.backtest import BacktestResult
from atp.domain.models.market import PriceHistory
from atp.domain.models.strategy import StrategyDefinition


class BacktestEngine(Protocol):
    """Runs a declarative strategy against price history.

    Synchronous by design (CPU-bound); callers offload to a worker thread.
    Implementations must be free of look-ahead bias: indicators may use only
    past bars, and orders fill no earlier than the next bar's open.
    """

    def run(
        self,
        strategy: StrategyDefinition,
        history: PriceHistory,
        *,
        initial_cash: float = 10_000.0,
        commission: float = 0.001,
    ) -> BacktestResult: ...
