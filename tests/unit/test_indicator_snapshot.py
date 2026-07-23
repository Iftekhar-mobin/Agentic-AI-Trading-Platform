"""Tests for the indicator snapshot builder and its classification rules."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from atp.domain.errors import InsufficientHistoryError
from atp.domain.models.analysis import MIN_HISTORY_BARS, SignalDirection
from atp.domain.models.market import Bar, BarInterval, PriceHistory
from atp.infrastructure.indicators import PandasIndicatorEngine


def make_history(closes: list[float], symbol: str = "TEST") -> PriceHistory:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    bars = tuple(
        Bar(
            timestamp=start + timedelta(days=i),
            open=close - 0.5,
            high=close + 1.0,
            low=close - 1.5,
            close=close,
            volume=1_000.0 + 10.0 * i,
        )
        for i, close in enumerate(closes)
    )
    return PriceHistory(symbol=symbol, interval=BarInterval.DAY_1, bars=bars)


def uptrend(n: int = 80) -> PriceHistory:
    return make_history([100.0 + i for i in range(n)])


def wavy_uptrend(n: int = 80) -> PriceHistory:
    """Rising but with pullbacks, so range-based indicators (StochRSI) stay defined."""
    return make_history([100.0 + 0.3 * i + (i % 7) for i in range(n)])


def downtrend(n: int = 80) -> PriceHistory:
    return make_history([200.0 - i for i in range(n)])


class TestPandasIndicatorEngine:
    def test_rejects_insufficient_history(self) -> None:
        with pytest.raises(InsufficientHistoryError, match=str(MIN_HISTORY_BARS)):
            PandasIndicatorEngine().compute_readings(uptrend(MIN_HISTORY_BARS - 1))

    def test_produces_all_core_readings(self) -> None:
        readings = PandasIndicatorEngine().compute_readings(wavy_uptrend())
        names = {reading.name for reading in readings}
        assert names == {
            "rsi_14",
            "stoch_rsi_14",
            "macd_12_26_9",
            "sma_trend",
            "ema_20_50",
            "bollinger_20_2",
            "vwap_window",
            "obv_trend",
            "atr_14",
        }

    def test_uptrend_classifications(self) -> None:
        readings = {r.name: r for r in PandasIndicatorEngine().compute_readings(uptrend())}
        # Trend and volume indicators confirm the uptrend...
        assert readings["sma_trend"].direction is SignalDirection.BULLISH
        assert readings["ema_20_50"].direction is SignalDirection.BULLISH
        assert readings["macd_12_26_9"].direction is SignalDirection.BULLISH
        assert readings["vwap_window"].direction is SignalDirection.BULLISH
        assert readings["obv_trend"].direction is SignalDirection.BULLISH
        # ...while a monotonic rise pegs RSI at 100 = overbought (bearish risk).
        assert readings["rsi_14"].direction is SignalDirection.BEARISH
        assert readings["rsi_14"].values["rsi"] == pytest.approx(100.0)
        # ATR is always volatility context, never directional.
        assert readings["atr_14"].direction is SignalDirection.NEUTRAL
        # RSI pinned at 100 has zero range, so StochRSI is undefined and dropped.
        assert "stoch_rsi_14" not in readings

    def test_downtrend_classifications(self) -> None:
        readings = {r.name: r for r in PandasIndicatorEngine().compute_readings(downtrend())}
        assert readings["sma_trend"].direction is SignalDirection.BEARISH
        assert readings["ema_20_50"].direction is SignalDirection.BEARISH
        assert readings["macd_12_26_9"].direction is SignalDirection.BEARISH
        assert readings["rsi_14"].direction is SignalDirection.BULLISH  # oversold

    def test_summaries_contain_values(self) -> None:
        readings = {r.name: r for r in PandasIndicatorEngine().compute_readings(uptrend())}
        sma_50 = readings["sma_trend"].values["sma_50"]
        assert f"{sma_50:.2f}" in readings["sma_trend"].summary

    def test_deterministic(self) -> None:
        first = PandasIndicatorEngine().compute_readings(uptrend())
        second = PandasIndicatorEngine().compute_readings(uptrend())
        assert first == second
