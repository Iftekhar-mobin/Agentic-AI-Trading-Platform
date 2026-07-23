"""Tests for the backtesting.py engine adapter on synthetic price data."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from atp.domain.errors import InsufficientHistoryError
from atp.domain.models.market import Bar, BarInterval, PriceHistory
from atp.domain.models.strategy import (
    Comparison,
    Condition,
    Operand,
    OperandKind,
    StrategyDefinition,
)
from atp.domain.strategy_presets import PRESETS
from atp.infrastructure.backtesting import BacktestingPyEngine


def make_history(closes: list[float]) -> PriceHistory:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    bars = tuple(
        Bar(
            timestamp=start + timedelta(days=i),
            open=close * 0.999,
            high=close * 1.01,
            low=close * 0.985,
            close=close,
            volume=1_000_000.0,
        )
        for i, close in enumerate(closes)
    )
    return PriceHistory(symbol="TEST", interval=BarInterval.DAY_1, bars=bars)


def v_shape(down: int = 80, up: int = 120) -> PriceHistory:
    """Decline then strong recovery: guarantees an EMA/SMA cross mid-series."""
    closes = [100.0 - 0.25 * i for i in range(down)]
    bottom = closes[-1]
    closes += [bottom + 0.5 * i for i in range(1, up + 1)]
    return make_history(closes)


ALWAYS_ENTER = StrategyDefinition(
    name="always_enter",
    entry=(
        Condition(
            left=Operand(kind=OperandKind.CLOSE),
            comparison=Comparison.GT,
            right=Operand(kind=OperandKind.CONSTANT, value=0.0),
        ),
    ),
    exit=(
        Condition(
            left=Operand(kind=OperandKind.CLOSE),
            comparison=Comparison.LT,
            right=Operand(kind=OperandKind.CONSTANT, value=0.0),
        ),
    ),
)


class TestBacktestingPyEngine:
    def test_rejects_insufficient_history(self) -> None:
        with pytest.raises(InsufficientHistoryError, match="ema_cross"):
            BacktestingPyEngine().run(PRESETS["ema_cross"], make_history([100.0] * 30))

    def test_ema_cross_trades_the_recovery(self) -> None:
        result = BacktestingPyEngine().run(PRESETS["ema_cross"], v_shape())

        assert result.bars == 200
        assert result.metrics.trades >= 1
        assert result.trade_log
        # The recovery is strong and sustained: the strategy must end profitable.
        assert result.final_equity > result.initial_cash
        assert result.metrics.total_return_pct > 0
        assert result.metrics.win_rate is not None

    def test_no_trades_before_indicator_warmup(self) -> None:
        """Look-ahead guard: entries require defined (post-warmup) indicator values."""
        result = BacktestingPyEngine().run(PRESETS["ema_cross"], v_shape())
        warmup_end = v_shape().bars[PRESETS["ema_cross"].warmup_bars].timestamp
        assert all(trade.entry_time > warmup_end for trade in result.trade_log)

    def test_signal_fills_on_next_bar_not_signal_bar(self) -> None:
        """Look-ahead guard: with an always-true entry, the fill is bar 1, not bar 0."""
        history = make_history([100.0 + i for i in range(30)])
        result = BacktestingPyEngine().run(ALWAYS_ENTER, history)

        assert result.metrics.trades == 1  # finalized at the end
        assert result.trade_log[0].entry_time >= history.bars[1].timestamp

    def test_rsi_reversion_runs(self) -> None:
        # Oscillating series with dips deep enough to push RSI below 30.
        closes = [
            100.0 + (15.0 if (i // 10) % 2 == 0 else -15.0) * ((i % 10) / 10) for i in range(150)
        ]
        result = BacktestingPyEngine().run(PRESETS["rsi_reversion"], make_history(closes))
        assert result.strategy.name == "rsi_reversion"
        assert result.metrics.max_drawdown_pct <= 0

    def test_result_embeds_strategy_and_window(self) -> None:
        history = v_shape()
        result = BacktestingPyEngine().run(PRESETS["sma_breakout"], history)
        assert result.strategy == PRESETS["sma_breakout"]
        assert result.start == history.bars[0].timestamp
        assert result.end == history.bars[-1].timestamp
        assert result.symbol == "TEST"
