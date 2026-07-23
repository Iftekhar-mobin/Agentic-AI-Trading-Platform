"""Golden-value tests for backtest metrics."""

from __future__ import annotations

import pandas as pd
import pytest

from atp.infrastructure.backtesting.metrics import compute_metrics

DAILY = 252.0


def equity(*values: float) -> pd.Series:
    return pd.Series([float(v) for v in values])


class TestEquityMetrics:
    def test_flat_equity(self) -> None:
        metrics = compute_metrics(equity(*([10_000.0] * 30)), [], DAILY)
        assert metrics.total_return_pct == 0.0
        assert metrics.max_drawdown_pct == 0.0
        assert metrics.sharpe is None  # zero variance: undefined, not zero
        assert metrics.sortino is None
        assert metrics.cagr_pct == pytest.approx(0.0)

    def test_known_drawdown(self) -> None:
        metrics = compute_metrics(equity(100, 120, 90, 130), [], DAILY)
        # Peak 120 -> trough 90 = -25%
        assert metrics.max_drawdown_pct == pytest.approx(-25.0)
        assert metrics.total_return_pct == pytest.approx(30.0)

    def test_steady_growth_has_positive_sharpe_and_undefined_sortino(self) -> None:
        curve = equity(*[10_000.0 * (1.001**i) for i in range(100)])
        metrics = compute_metrics(curve, [], DAILY)
        assert metrics.sharpe is not None
        assert metrics.sharpe > 0
        # No losing periods: downside deviation is 0, sortino undefined.
        assert metrics.sortino is None
        assert metrics.cagr_pct is not None
        assert metrics.cagr_pct > 0

    def test_volatile_curve_has_sortino(self) -> None:
        values = [10_000.0]
        for i in range(60):
            values.append(values[-1] * (1.01 if i % 2 == 0 else 0.995))
        metrics = compute_metrics(equity(*values), [], DAILY)
        assert metrics.sortino is not None
        assert metrics.sharpe is not None

    def test_total_loss_has_no_cagr(self) -> None:
        metrics = compute_metrics(equity(10_000, 0.0), [], DAILY)
        assert metrics.cagr_pct is None
        assert metrics.max_drawdown_pct == pytest.approx(-100.0)


class TestTradeMetrics:
    def test_no_trades(self) -> None:
        metrics = compute_metrics(equity(100, 100), [], DAILY)
        assert metrics.trades == 0
        assert metrics.win_rate is None
        assert metrics.profit_factor is None
        assert metrics.expectancy_pct is None

    def test_known_trade_stats(self) -> None:
        metrics = compute_metrics(equity(100, 120), [10.0, -5.0, 15.0], DAILY)
        assert metrics.trades == 3
        assert metrics.win_rate == pytest.approx(2 / 3)
        assert metrics.profit_factor == pytest.approx(25.0 / 5.0)
        assert metrics.expectancy_pct == pytest.approx(20.0 / 3.0)

    def test_profit_factor_undefined_without_losses(self) -> None:
        metrics = compute_metrics(equity(100, 120), [10.0, 5.0], DAILY)
        assert metrics.profit_factor is None
        assert metrics.win_rate == 1.0
