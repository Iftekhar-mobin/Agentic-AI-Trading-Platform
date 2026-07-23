"""Pure indicator functions over pandas Series.

Every number the platform reports originates in deterministic, golden-tested
code like this — LLMs interpret these values but never compute them.

Conventions:
- EMA uses span smoothing (alpha = 2/(n+1)); RSI and ATR use Wilder's
  smoothing (alpha = 1/n), matching common charting platforms.
- Warmup periods yield NaN (via ``min_periods``) so unstable early values can
  never be mistaken for signals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(close: pd.Series, period: int) -> pd.Series:
    return close.rolling(period).mean()


def ema(close: pd.Series, period: int) -> pd.Series:
    return close.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    result = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    # A perfectly flat series gives 0/0 = NaN; define it as neutral 50.
    return result.mask((avg_gain == 0.0) & (avg_loss == 0.0), 50.0)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "histogram": macd_line - signal_line}
    )


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return true_range.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def bollinger(close: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    middle = sma(close, period)
    std = close.rolling(period).std(ddof=0)  # population std, per TA convention
    return pd.DataFrame(
        {"middle": middle, "upper": middle + num_std * std, "lower": middle - num_std * std}
    )


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction: pd.Series = np.sign(close.diff()).fillna(0.0)
    result: pd.Series = (direction * volume).cumsum()
    return result


def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """Cumulative VWAP over the whole window (session-agnostic)."""
    typical = (high + low + close) / 3.0
    cumulative_volume = volume.cumsum()
    return (typical * volume).cumsum() / cumulative_volume.mask(cumulative_volume == 0.0)


def stochastic_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Position of RSI within its recent range, in [0, 1]. NaN when the range is zero."""
    rsi_series = rsi(close, period)
    lowest = rsi_series.rolling(period).min()
    highest = rsi_series.rolling(period).max()
    result: pd.Series = (rsi_series - lowest) / (highest - lowest)
    return result
