"""Builds classified indicator readings from price history.

Classification rules here are deliberately simple, deterministic conventions
(RSI 70/30, MACD-vs-signal, price-vs-MA, %B bands). The LLM agent layers
interpretation on top; it never changes these classifications.
"""

from __future__ import annotations

import math

import pandas as pd

from atp.domain.errors import InsufficientHistoryError
from atp.domain.models.analysis import MIN_HISTORY_BARS, IndicatorReading, SignalDirection
from atp.domain.models.market import PriceHistory
from atp.infrastructure.indicators import engine

BULLISH = SignalDirection.BULLISH
BEARISH = SignalDirection.BEARISH
NEUTRAL = SignalDirection.NEUTRAL


def to_frame(history: PriceHistory) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [bar.open for bar in history.bars],
            "high": [bar.high for bar in history.bars],
            "low": [bar.low for bar in history.bars],
            "close": [bar.close for bar in history.bars],
            "volume": [bar.volume for bar in history.bars],
        },
        index=pd.DatetimeIndex([bar.timestamp for bar in history.bars], name="ts"),
    )


class PandasIndicatorEngine:
    """IndicatorEngine implementation backed by the pandas engine functions."""

    def compute_readings(self, history: PriceHistory) -> tuple[IndicatorReading, ...]:
        if len(history) < MIN_HISTORY_BARS:
            msg = (
                f"{history.symbol}/{history.interval.value}: {len(history)} bars available, "
                f"{MIN_HISTORY_BARS} required for an indicator snapshot"
            )
            raise InsufficientHistoryError(msg)

        frame = to_frame(history)
        close = frame["close"]
        latest_close = float(close.iloc[-1])

        candidates = [
            self._rsi(close),
            self._stochastic_rsi(close),
            self._macd(close),
            self._moving_averages(close, latest_close),
            self._ema_cross(close),
            self._bollinger(close, latest_close),
            self._vwap(frame, latest_close),
            self._obv(frame),
            self._atr(frame, latest_close),
        ]
        return tuple(reading for reading in candidates if reading is not None)

    @staticmethod
    def _rsi(close: pd.Series) -> IndicatorReading | None:
        value = float(engine.rsi(close, 14).iloc[-1])
        if math.isnan(value):
            return None
        if value >= 70:
            direction, note = BEARISH, "overbought - pullback risk"
        elif value <= 30:
            direction, note = BULLISH, "oversold - bounce potential"
        else:
            direction, note = NEUTRAL, "in neutral range"
        return IndicatorReading(
            name="rsi_14",
            values={"rsi": round(value, 2)},
            direction=direction,
            summary=f"RSI(14) is {value:.1f} - {note}",
        )

    @staticmethod
    def _stochastic_rsi(close: pd.Series) -> IndicatorReading | None:
        value = float(engine.stochastic_rsi(close, 14).iloc[-1])
        if math.isnan(value):
            return None
        if value >= 0.8:
            direction, note = BEARISH, "RSI near the top of its recent range"
        elif value <= 0.2:
            direction, note = BULLISH, "RSI near the bottom of its recent range"
        else:
            direction, note = NEUTRAL, "RSI mid-range"
        return IndicatorReading(
            name="stoch_rsi_14",
            values={"stoch_rsi": round(value, 3)},
            direction=direction,
            summary=f"StochRSI(14) is {value:.2f} - {note}",
        )

    @staticmethod
    def _macd(close: pd.Series) -> IndicatorReading | None:
        frame = engine.macd(close)
        macd_line = float(frame["macd"].iloc[-1])
        signal_line = float(frame["signal"].iloc[-1])
        histogram = float(frame["histogram"].iloc[-1])
        if math.isnan(macd_line) or math.isnan(signal_line):
            return None
        direction = BULLISH if macd_line > signal_line else BEARISH
        relation = "above" if direction is BULLISH else "below"
        return IndicatorReading(
            name="macd_12_26_9",
            values={
                "macd": round(macd_line, 4),
                "signal": round(signal_line, 4),
                "histogram": round(histogram, 4),
            },
            direction=direction,
            summary=f"MACD {macd_line:.3f} is {relation} its signal line {signal_line:.3f}",
        )

    @staticmethod
    def _moving_averages(close: pd.Series, latest_close: float) -> IndicatorReading | None:
        sma_50 = float(engine.sma(close, 50).iloc[-1])
        if math.isnan(sma_50):
            return None
        values = {"sma_50": round(sma_50, 2)}
        above = latest_close > sma_50
        summary = (
            f"Close {latest_close:.2f} is {'above' if above else 'below'} SMA(50) {sma_50:.2f}"
        )
        if len(close) >= 200:
            sma_200 = float(engine.sma(close, 200).iloc[-1])
            values["sma_200"] = round(sma_200, 2)
            summary += f"; SMA(200) is {sma_200:.2f}"
        return IndicatorReading(
            name="sma_trend",
            values=values,
            direction=BULLISH if above else BEARISH,
            summary=summary,
        )

    @staticmethod
    def _ema_cross(close: pd.Series) -> IndicatorReading | None:
        ema_20 = float(engine.ema(close, 20).iloc[-1])
        ema_50 = float(engine.ema(close, 50).iloc[-1])
        if math.isnan(ema_20) or math.isnan(ema_50):
            return None
        direction = BULLISH if ema_20 > ema_50 else BEARISH
        relation = "above" if direction is BULLISH else "below"
        return IndicatorReading(
            name="ema_20_50",
            values={"ema_20": round(ema_20, 2), "ema_50": round(ema_50, 2)},
            direction=direction,
            summary=f"EMA(20) {ema_20:.2f} is {relation} EMA(50) {ema_50:.2f}",
        )

    @staticmethod
    def _bollinger(close: pd.Series, latest_close: float) -> IndicatorReading | None:
        bands = engine.bollinger(close, 20, 2.0)
        upper = float(bands["upper"].iloc[-1])
        lower = float(bands["lower"].iloc[-1])
        middle = float(bands["middle"].iloc[-1])
        if math.isnan(upper) or math.isnan(lower):
            return None
        band_width = upper - lower
        percent_b = (latest_close - lower) / band_width if band_width > 0 else 0.5
        if percent_b > 1.0:
            direction, note = BEARISH, "close above the upper band (stretched)"
        elif percent_b < 0.0:
            direction, note = BULLISH, "close below the lower band (stretched)"
        else:
            direction, note = NEUTRAL, "close within the bands"
        return IndicatorReading(
            name="bollinger_20_2",
            values={
                "upper": round(upper, 2),
                "middle": round(middle, 2),
                "lower": round(lower, 2),
                "percent_b": round(percent_b, 3),
            },
            direction=direction,
            summary=f"%B is {percent_b:.2f} - {note}",
        )

    @staticmethod
    def _vwap(frame: pd.DataFrame, latest_close: float) -> IndicatorReading | None:
        value = float(
            engine.vwap(frame["high"], frame["low"], frame["close"], frame["volume"]).iloc[-1]
        )
        if math.isnan(value):
            return None
        above = latest_close > value
        return IndicatorReading(
            name="vwap_window",
            values={"vwap": round(value, 2)},
            direction=BULLISH if above else BEARISH,
            summary=(
                f"Close {latest_close:.2f} is {'above' if above else 'below'} "
                f"window VWAP {value:.2f}"
            ),
        )

    @staticmethod
    def _obv(frame: pd.DataFrame) -> IndicatorReading | None:
        obv_series = engine.obv(frame["close"], frame["volume"])
        obv_now = float(obv_series.iloc[-1])
        obv_avg = float(engine.sma(obv_series, 20).iloc[-1])
        if math.isnan(obv_avg):
            return None
        rising = obv_now > obv_avg
        return IndicatorReading(
            name="obv_trend",
            values={"obv": round(obv_now, 0), "obv_sma_20": round(obv_avg, 0)},
            direction=BULLISH if rising else BEARISH,
            summary=f"OBV is {'above' if rising else 'below'} its 20-bar average - "
            f"volume {'confirms buying' if rising else 'leans to selling'}",
        )

    @staticmethod
    def _atr(frame: pd.DataFrame, latest_close: float) -> IndicatorReading | None:
        value = float(engine.atr(frame["high"], frame["low"], frame["close"], 14).iloc[-1])
        if math.isnan(value):
            return None
        atr_pct = 100.0 * value / latest_close
        return IndicatorReading(
            name="atr_14",
            values={"atr": round(value, 3), "atr_pct": round(atr_pct, 2)},
            direction=NEUTRAL,
            summary=f"ATR(14) is {value:.2f} ({atr_pct:.1f}% of price) - volatility context",
        )
