"""Tests for Monte Carlo trade resampling."""

from __future__ import annotations

import pytest

from atp.domain.services import monte_carlo_trades


class TestMonteCarloTrades:
    def test_too_few_trades_returns_none(self) -> None:
        assert monte_carlo_trades([]) is None
        assert monte_carlo_trades([5.0, -2.0]) is None

    def test_identical_trades_have_zero_spread(self) -> None:
        summary = monte_carlo_trades([10.0, 10.0, 10.0])
        assert summary is not None
        expected = (1.1**3 - 1) * 100
        assert summary.return_p5_pct == pytest.approx(expected)
        assert summary.return_p50_pct == pytest.approx(expected)
        assert summary.return_p95_pct == pytest.approx(expected)
        assert summary.probability_of_loss == 0.0
        assert summary.max_drawdown_p50_pct == 0.0

    def test_deterministic_with_seed(self) -> None:
        returns = [12.0, -8.0, 5.0, -3.0, 20.0, -6.0]
        first = monte_carlo_trades(returns, seed=7)
        second = monte_carlo_trades(returns, seed=7)
        assert first == second
        different = monte_carlo_trades(returns, seed=8)
        assert different != first

    def test_percentiles_are_ordered(self) -> None:
        summary = monte_carlo_trades([12.0, -8.0, 5.0, -3.0, 20.0, -6.0])
        assert summary is not None
        assert summary.return_p5_pct <= summary.return_p50_pct <= summary.return_p95_pct
        # p95 drawdown is the bad tail: at least as negative as the median.
        assert summary.max_drawdown_p95_pct <= summary.max_drawdown_p50_pct
        assert 0.0 <= summary.probability_of_loss <= 1.0

    def test_losing_strategy_shows_loss_probability(self) -> None:
        summary = monte_carlo_trades([-5.0, -3.0, 1.0, -4.0, -6.0])
        assert summary is not None
        assert summary.probability_of_loss > 0.9
        assert summary.return_p50_pct < 0
