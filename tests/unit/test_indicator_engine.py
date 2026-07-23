"""Golden-value tests for the deterministic indicator engine.

Expected values are hand-computed from the definitions, so these tests pin the
exact numerical conventions (EMA span smoothing, Wilder RSI/ATR, population
std for Bollinger).
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from atp.infrastructure.indicators import engine


def series(*values: float) -> pd.Series:
    return pd.Series([float(v) for v in values])


class TestSMA:
    def test_simple_average(self) -> None:
        result = engine.sma(series(1, 2, 3, 4, 5), 3)
        assert math.isnan(result.iloc[0])
        assert math.isnan(result.iloc[1])
        assert list(result.iloc[2:]) == [2.0, 3.0, 4.0]


class TestEMA:
    def test_recursive_definition(self) -> None:
        # alpha = 2/(3+1) = 0.5: y0=2, y1=0.5*4+0.5*2=3, y2=0.5*8+0.5*3=5.5
        result = engine.ema(series(2, 4, 8), 3)
        assert math.isnan(result.iloc[0])  # warmup masked by min_periods
        assert math.isnan(result.iloc[1])
        assert result.iloc[2] == 5.5


class TestRSI:
    def test_all_gains_is_100(self) -> None:
        result = engine.rsi(series(*range(1, 31)), 14)
        assert result.iloc[-1] == pytest.approx(100.0)

    def test_all_losses_is_0(self) -> None:
        result = engine.rsi(series(*range(100, 40, -2)), 14)
        assert result.iloc[-1] == pytest.approx(0.0)

    def test_flat_series_is_neutral_50(self) -> None:
        result = engine.rsi(series(*([100.0] * 30)), 14)
        assert result.iloc[-1] == pytest.approx(50.0)

    def test_warmup_is_nan(self) -> None:
        result = engine.rsi(series(*range(1, 31)), 14)
        assert result.iloc[:14].isna().all()


class TestMACD:
    def test_flat_series_is_zero(self) -> None:
        result = engine.macd(series(*([50.0] * 40)))
        assert result["macd"].iloc[-1] == pytest.approx(0.0)
        assert result["signal"].iloc[-1] == pytest.approx(0.0)
        assert result["histogram"].iloc[-1] == pytest.approx(0.0)

    def test_uptrend_is_positive(self) -> None:
        result = engine.macd(series(*range(1, 61)))
        assert result["macd"].iloc[-1] > 0
        assert result["histogram"].iloc[-1] == pytest.approx(
            result["macd"].iloc[-1] - result["signal"].iloc[-1]
        )


class TestATR:
    def test_constant_range_bars(self) -> None:
        n = 20
        high = series(*([105.0] * n))
        low = series(*([100.0] * n))
        close = series(*([102.0] * n))
        result = engine.atr(high, low, close, 14)
        assert result.iloc[-1] == pytest.approx(5.0)


class TestBollinger:
    def test_constant_series_collapses_bands(self) -> None:
        result = engine.bollinger(series(*([100.0] * 25)), 20, 2.0)
        assert result["upper"].iloc[-1] == pytest.approx(100.0)
        assert result["middle"].iloc[-1] == pytest.approx(100.0)
        assert result["lower"].iloc[-1] == pytest.approx(100.0)

    def test_known_window(self) -> None:
        # Window [1..5]: mean 3, population std = sqrt(2)
        result = engine.bollinger(series(1, 2, 3, 4, 5), 5, 2.0)
        assert result["middle"].iloc[-1] == pytest.approx(3.0)
        assert result["upper"].iloc[-1] == pytest.approx(3.0 + 2.0 * math.sqrt(2.0))


class TestOBV:
    def test_accumulates_signed_volume(self) -> None:
        close = series(10, 11, 10, 10, 12)
        volume = series(100, 200, 300, 400, 500)
        result = engine.obv(close, volume)
        # +200 (up), -300 (down), +0 (flat), +500 (up)
        assert list(result) == [0.0, 200.0, -100.0, -100.0, 400.0]


class TestVWAP:
    def test_single_bar_is_typical_price(self) -> None:
        result = engine.vwap(series(105), series(95), series(100), series(1000))
        assert result.iloc[-1] == pytest.approx(100.0)

    def test_weights_by_volume(self) -> None:
        # bar1 typical 100 @ vol 1000, bar2 typical 200 @ vol 3000 -> 175
        result = engine.vwap(
            series(105, 210), series(95, 190), series(100, 200), series(1000, 3000)
        )
        assert result.iloc[-1] == pytest.approx(175.0)


class TestStochasticRSI:
    def test_bounded_zero_to_one(self) -> None:
        closes = [100 + ((i * 7) % 13) - 6 for i in range(60)]
        result = engine.stochastic_rsi(series(*closes), 14).dropna()
        assert not result.empty
        assert ((result >= 0) & (result <= 1)).all()
