"""Golden-value tests for deterministic market-regime tagging."""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import pytest

from atp.domain.errors import InsufficientHistoryError
from atp.domain.models.market import Bar, BarInterval, PriceHistory
from atp.domain.models.memory import RegimeTrend, RegimeVolatility
from atp.domain.services.regime_detection import MIN_REGIME_BARS, detect_regime

START = datetime(2026, 1, 1, tzinfo=UTC)


def history(closes: list[float], symbol: str = "AAPL") -> PriceHistory:
    bars = tuple(
        Bar(
            timestamp=START + timedelta(days=index),
            open=close,
            high=close * 1.01,
            low=close * 0.99,
            close=close,
            volume=1_000.0,
        )
        for index, close in enumerate(closes)
    )
    return PriceHistory(symbol=symbol, interval=BarInterval.DAY_1, bars=bars)


def series(count: int, price: Callable[[int], float]) -> list[float]:
    return [price(index) for index in range(count)]


def test_steady_rise_is_an_uptrend() -> None:
    regime = detect_regime(history(series(150, lambda i: 100.0 * 1.004**i)))
    assert regime.trend is RegimeTrend.UPTREND


def test_steady_fall_is_a_downtrend() -> None:
    regime = detect_regime(history(series(150, lambda i: 300.0 * 0.996**i)))
    assert regime.trend is RegimeTrend.DOWNTREND


def test_flat_market_is_sideways() -> None:
    regime = detect_regime(history(series(150, lambda i: 100.0 + 0.5 * math.sin(i / 4))))
    assert regime.trend is RegimeTrend.SIDEWAYS


def test_a_barely_crossed_average_is_not_a_trend() -> None:
    """Position and slope must agree, so noise near a crossover stays sideways."""
    closes = series(150, lambda i: 100.0 + 0.002 * i + 0.3 * math.sin(i / 3))
    assert detect_regime(history(closes)).trend is RegimeTrend.SIDEWAYS


def test_volatility_expansion_is_detected() -> None:
    """Calm for 130 bars, then a violent 20 — the recent window must dominate."""
    calm = series(130, lambda i: 100.0 + 0.05 * math.sin(i / 3))
    wild = [100.0 * (1.08 if index % 2 else 0.93) for index in range(20)]
    regime = detect_regime(history(calm + wild))

    assert regime.volatility is RegimeVolatility.VOLATILE
    assert regime.metrics["vol_ratio"] > 1.3


def test_volatility_contraction_is_detected() -> None:
    wild = [100.0 * (1.06 if index % 2 else 0.95) for index in range(130)]
    calm = [100.0 + 0.02 * index for index in range(20)]
    regime = detect_regime(history(wild + calm))

    assert regime.volatility is RegimeVolatility.CALM
    assert regime.metrics["vol_ratio"] < 0.7


def test_ordinary_volatility_reads_normal() -> None:
    closes = series(150, lambda i: 100.0 * 1.002**i + 0.4 * math.sin(i / 2))
    assert detect_regime(history(closes)).volatility is RegimeVolatility.NORMAL


def test_label_combines_both_axes() -> None:
    regime = detect_regime(history(series(150, lambda i: 100.0 * 1.004**i)))
    assert regime.label == f"{regime.trend.value}/{regime.volatility.value}"
    assert regime.label.count("/") == 1


def test_metrics_expose_the_numbers_behind_the_label() -> None:
    regime = detect_regime(history(series(150, lambda i: 100.0 * 1.004**i)))
    assert set(regime.metrics) == {
        "close",
        "sma_fast",
        "sma_slow",
        "slow_slope_pct_per_bar",
        "realized_vol",
        "baseline_vol",
        "vol_ratio",
    }
    assert regime.metrics["sma_fast"] > regime.metrics["sma_slow"]


def test_as_of_is_the_last_bar() -> None:
    closes = series(150, lambda i: 100.0 * 1.004**i)
    result = history(closes)
    assert detect_regime(result).as_of == result.bars[-1].timestamp


def test_short_history_is_rejected() -> None:
    closes = series(MIN_REGIME_BARS - 1, lambda i: 100.0 + i)
    with pytest.raises(InsufficientHistoryError, match="required to tag a regime"):
        detect_regime(history(closes))


def test_detection_is_reproducible() -> None:
    closes = series(150, lambda i: 100.0 * 1.003**i + math.sin(i))
    first = detect_regime(history(closes))
    second = detect_regime(history(closes))
    assert first == second
