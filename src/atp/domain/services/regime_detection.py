"""Deterministic market-regime tagging.

A remembered decision is only reusable next to the conditions it was made in:
"this breakout setup worked" means one thing in a calm uptrend and another in a
volatile chop. Regime tagging is therefore part of writing memory, not an
afterthought at recall time.

Both axes are computed from price alone, with fixed rules:

- **Trend** — the fast SMA against the slow SMA, plus the slow SMA's own slope
  measured in percent per bar. Requiring agreement between position and slope
  is what keeps a flat market with a hair's-breadth crossover from being
  labelled a trend.
- **Volatility** — recent realized volatility against the longer baseline of
  the same series, so the label is relative to the instrument rather than to a
  hardcoded percentage that would call every biotech "volatile".

Pure stdlib on purpose: this is domain logic, and it must produce identical
labels wherever it runs.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from itertools import pairwise

from atp.domain.errors import InsufficientHistoryError
from atp.domain.models.market import PriceHistory
from atp.domain.models.memory import RegimeTag, RegimeTrend, RegimeVolatility

FAST_WINDOW = 20
SLOW_WINDOW = 50
VOLATILITY_WINDOW = 20
BASELINE_WINDOW = 100

MIN_REGIME_BARS = SLOW_WINDOW + 5
"""Enough bars for a slow SMA plus a slope measured across it."""

SLOPE_WINDOW = 10
"""Bars over which the slow SMA's slope is measured."""

TREND_SLOPE_PCT = 0.05
"""Minimum |slope| in percent per bar before a trend is called (5% over 100 bars)."""

VOLATILE_RATIO = 1.3
"""Recent vol above this multiple of the baseline reads as volatile."""

CALM_RATIO = 0.7
"""Recent vol below this multiple of the baseline reads as calm."""


def _sma(values: Sequence[float], window: int) -> float:
    return sum(values[-window:]) / window


def _realized_volatility(closes: Sequence[float], window: int) -> float:
    """Annualized stdev of log returns over the last ``window`` returns."""
    sample = closes[-(window + 1) :]
    returns = [
        math.log(later / earlier)
        for earlier, later in pairwise(sample)
        if earlier > 0 and later > 0
    ]
    if len(returns) < 2:
        return 0.0
    return statistics.stdev(returns) * math.sqrt(252)


def detect_regime(history: PriceHistory) -> RegimeTag:
    """Label the market conditions at the end of ``history``."""
    if len(history) < MIN_REGIME_BARS:
        msg = (
            f"{history.symbol}/{history.interval.value}: {len(history)} bars available, "
            f"{MIN_REGIME_BARS} required to tag a regime"
        )
        raise InsufficientHistoryError(msg)

    closes = [bar.close for bar in history.bars]
    latest = history.bars[-1]

    fast = _sma(closes, FAST_WINDOW)
    slow = _sma(closes, SLOW_WINDOW)
    previous_slow = _sma(closes[:-SLOPE_WINDOW], SLOW_WINDOW)
    slope_pct = (
        (slow - previous_slow) / previous_slow * 100 / SLOPE_WINDOW if previous_slow > 0 else 0.0
    )

    trend = _classify_trend(fast, slow, slope_pct)

    recent_vol = _realized_volatility(closes, VOLATILITY_WINDOW)
    baseline_vol = _realized_volatility(closes, min(BASELINE_WINDOW, len(closes) - 1))
    vol_ratio = recent_vol / baseline_vol if baseline_vol > 0 else 1.0
    volatility = _classify_volatility(vol_ratio)

    return RegimeTag(
        trend=trend,
        volatility=volatility,
        as_of=latest.timestamp,
        metrics={
            "close": round(latest.close, 4),
            "sma_fast": round(fast, 4),
            "sma_slow": round(slow, 4),
            "slow_slope_pct_per_bar": round(slope_pct, 4),
            "realized_vol": round(recent_vol, 4),
            "baseline_vol": round(baseline_vol, 4),
            "vol_ratio": round(vol_ratio, 4),
        },
    )


def _classify_trend(fast: float, slow: float, slope_pct: float) -> RegimeTrend:
    """Position and slope must agree; disagreement is a turning market, i.e. sideways."""
    if fast > slow and slope_pct > TREND_SLOPE_PCT:
        return RegimeTrend.UPTREND
    if fast < slow and slope_pct < -TREND_SLOPE_PCT:
        return RegimeTrend.DOWNTREND
    return RegimeTrend.SIDEWAYS


def _classify_volatility(vol_ratio: float) -> RegimeVolatility:
    if vol_ratio >= VOLATILE_RATIO:
        return RegimeVolatility.VOLATILE
    if vol_ratio <= CALM_RATIO:
        return RegimeVolatility.CALM
    return RegimeVolatility.NORMAL
