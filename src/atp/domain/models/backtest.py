"""Backtest result models."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from atp.domain.models.market import BarInterval
from atp.domain.models.strategy import StrategyDefinition


class TradeRecord(BaseModel):
    """One closed (or force-finalized) trade; kept for audit and future learning."""

    model_config = ConfigDict(frozen=True)

    entry_time: datetime
    exit_time: datetime
    return_pct: float


class BacktestMetrics(BaseModel):
    """Metrics computed deterministically from the equity curve and trade list.

    ``None`` means "undefined for this run" (e.g. Sharpe with zero return
    variance, profit factor with no losing trades) — never silently 0.
    """

    model_config = ConfigDict(frozen=True)

    trades: int
    win_rate: float | None
    profit_factor: float | None
    expectancy_pct: float | None
    total_return_pct: float
    cagr_pct: float | None
    sharpe: float | None
    sortino: float | None
    max_drawdown_pct: float


class BacktestResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    strategy: StrategyDefinition  # embedded in full, so results are reproducible
    symbol: str
    interval: BarInterval
    start: datetime
    end: datetime
    bars: int
    initial_cash: float
    final_equity: float
    metrics: BacktestMetrics
    trade_log: tuple[TradeRecord, ...]
