"""Deterministic performance metrics from an equity curve and trade returns.

Computed in our own tested code rather than trusting library stats, so the
numbers are identical regardless of which backtest engine produced the run.
``None`` means a metric is undefined for the run — never silently 0.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import pandas as pd

from atp.domain.models.backtest import BacktestMetrics
from atp.domain.models.market import BarInterval

# Trading periods per year, for annualizing Sharpe/Sortino/CAGR.
# Intraday values assume US equity hours (6.5h/day, 252 days).
PERIODS_PER_YEAR: dict[BarInterval, float] = {
    BarInterval.MIN_1: 252 * 390.0,
    BarInterval.MIN_5: 252 * 78.0,
    BarInterval.MIN_15: 252 * 26.0,
    BarInterval.HOUR_1: 252 * 6.5,
    BarInterval.DAY_1: 252.0,
    BarInterval.WEEK_1: 52.0,
}


def compute_metrics(
    equity: pd.Series,
    trade_returns_pct: Sequence[float],
    periods_per_year: float,
) -> BacktestMetrics:
    returns = equity.pct_change().dropna()
    initial = float(equity.iloc[0])
    final = float(equity.iloc[-1])
    total_return = final / initial - 1.0

    drawdown = equity / equity.cummax() - 1.0
    max_drawdown_pct = float(drawdown.min()) * 100.0

    return BacktestMetrics(
        trades=len(trade_returns_pct),
        win_rate=_win_rate(trade_returns_pct),
        profit_factor=_profit_factor(trade_returns_pct),
        expectancy_pct=_mean(trade_returns_pct),
        total_return_pct=total_return * 100.0,
        cagr_pct=_cagr(total_return, len(returns), periods_per_year),
        sharpe=_sharpe(returns, periods_per_year),
        sortino=_sortino(returns, periods_per_year),
        max_drawdown_pct=max_drawdown_pct,
    )


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _win_rate(trade_returns_pct: Sequence[float]) -> float | None:
    if not trade_returns_pct:
        return None
    wins = sum(1 for r in trade_returns_pct if r > 0)
    return wins / len(trade_returns_pct)


def _profit_factor(trade_returns_pct: Sequence[float]) -> float | None:
    gross_profit = sum(r for r in trade_returns_pct if r > 0)
    gross_loss = -sum(r for r in trade_returns_pct if r < 0)
    if gross_loss == 0:
        return None  # undefined without losses (would be infinite)
    return gross_profit / gross_loss


def _cagr(total_return: float, periods: int, periods_per_year: float) -> float | None:
    if periods == 0 or total_return <= -1.0:
        return None
    years = periods / periods_per_year
    if years <= 0:
        return None
    return (float((1.0 + total_return) ** (1.0 / years)) - 1.0) * 100.0


def _sharpe(returns: pd.Series, periods_per_year: float) -> float | None:
    if len(returns) < 2:
        return None
    std = float(returns.std())
    if std == 0 or math.isnan(std):
        return None
    return float(returns.mean()) / std * math.sqrt(periods_per_year)


def _sortino(returns: pd.Series, periods_per_year: float) -> float | None:
    if len(returns) < 2:
        return None
    downside = returns.clip(upper=0.0)
    downside_dev = math.sqrt(float((downside**2).mean()))
    if downside_dev == 0:
        return None  # no losing periods: sortino is undefined (infinite)
    return float(returns.mean()) / downside_dev * math.sqrt(periods_per_year)
