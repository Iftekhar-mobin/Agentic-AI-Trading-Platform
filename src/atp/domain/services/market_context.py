"""Deterministic market-relative measurements against a benchmark.

Pure stdlib, in the domain, for the same reason as the other rule engines: the
numbers must be reproducible and the LLM must not be the thing computing them.

The one piece of care worth flagging is alignment. Two symbols do not
necessarily share a bar history — different listing dates, halts, holidays on
one exchange and not the other — so every statistic here is computed on the
*intersection* of timestamps. Comparing the last 63 bars of one series to the
last 63 of another without aligning them is a very easy way to produce
confident nonsense about a stock that was suspended for a week.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import pairwise

from atp.domain.errors import InsufficientHistoryError
from atp.domain.models.analysis import SignalDirection
from atp.domain.models.market import PriceHistory
from atp.domain.models.research import MIN_RESEARCH_BARS, MarketContextReading

BULLISH = SignalDirection.BULLISH
BEARISH = SignalDirection.BEARISH
NEUTRAL = SignalDirection.NEUTRAL

RELATIVE_STRENGTH_WINDOWS: dict[str, int] = {
    "1m": 21,
    "3m": 63,
    "12m": 252,
}
"""Bar counts named for their daily-timeframe equivalent."""

STATS_WINDOW = 63
"""Bars used for beta and correlation — a quarter of daily data."""

BENCHMARK_TREND_WINDOW = 50

RELATIVE_STRENGTH_THRESHOLD_PCT = 2.0
"""Outperformance below this is noise, not a signal."""


def align(
    symbol_history: PriceHistory, benchmark_history: PriceHistory
) -> tuple[list[float], list[float]]:
    """Return closes for the timestamps both series share, oldest first."""
    benchmark_by_time = {bar.timestamp: bar.close for bar in benchmark_history.bars}
    symbol_closes: list[float] = []
    benchmark_closes: list[float] = []
    for bar in symbol_history.bars:
        benchmark_close = benchmark_by_time.get(bar.timestamp)
        if benchmark_close is not None:
            symbol_closes.append(bar.close)
            benchmark_closes.append(benchmark_close)
    return symbol_closes, benchmark_closes


def _returns(closes: Sequence[float]) -> list[float]:
    return [(later / earlier) - 1.0 for earlier, later in pairwise(closes) if earlier > 0]


def _window_return_pct(closes: Sequence[float], window: int) -> float | None:
    if len(closes) < window + 1:
        return None
    start = closes[-(window + 1)]
    return (closes[-1] / start - 1.0) * 100 if start > 0 else None


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def analyze(
    symbol_history: PriceHistory, benchmark_history: PriceHistory
) -> tuple[MarketContextReading, ...]:
    """Compute every market-relative reading the overlap supports."""
    symbol_closes, benchmark_closes = align(symbol_history, benchmark_history)
    if len(symbol_closes) < MIN_RESEARCH_BARS:
        msg = (
            f"{symbol_history.symbol} vs {benchmark_history.symbol}: "
            f"{len(symbol_closes)} overlapping bars, {MIN_RESEARCH_BARS} required "
            "for market research"
        )
        raise InsufficientHistoryError(msg)

    readings: list[MarketContextReading | None] = [
        *(
            _relative_strength(label, window, symbol_closes, benchmark_closes)
            for label, window in RELATIVE_STRENGTH_WINDOWS.items()
        ),
        _beta_and_correlation(symbol_closes, benchmark_closes, benchmark_history.symbol),
        _benchmark_trend(benchmark_closes, benchmark_history.symbol),
        _drawdown_comparison(symbol_closes, benchmark_closes, benchmark_history.symbol),
    ]
    return tuple(reading for reading in readings if reading is not None)


def _relative_strength(
    label: str,
    window: int,
    symbol_closes: Sequence[float],
    benchmark_closes: Sequence[float],
) -> MarketContextReading | None:
    symbol_return = _window_return_pct(symbol_closes, window)
    benchmark_return = _window_return_pct(benchmark_closes, window)
    if symbol_return is None or benchmark_return is None:
        return None

    spread = symbol_return - benchmark_return
    if spread > RELATIVE_STRENGTH_THRESHOLD_PCT:
        direction, note = BULLISH, "outperforming the market"
    elif spread < -RELATIVE_STRENGTH_THRESHOLD_PCT:
        direction, note = BEARISH, "lagging the market"
    else:
        direction, note = NEUTRAL, "moving with the market"

    return MarketContextReading(
        name=f"relative_strength_{label}",
        values={
            "symbol_return_pct": round(symbol_return, 2),
            "benchmark_return_pct": round(benchmark_return, 2),
            "spread_pct": round(spread, 2),
        },
        direction=direction,
        summary=(
            f"Over {window} bars the symbol returned {symbol_return:+.1f}% against the "
            f"benchmark's {benchmark_return:+.1f}% ({spread:+.1f}pp) - {note}"
        ),
    )


def _beta_and_correlation(
    symbol_closes: Sequence[float], benchmark_closes: Sequence[float], benchmark: str
) -> MarketContextReading | None:
    """Beta and correlation are context, not a direction — they qualify other signals."""
    symbol_returns = _returns(symbol_closes[-(STATS_WINDOW + 1) :])
    benchmark_returns = _returns(benchmark_closes[-(STATS_WINDOW + 1) :])
    if len(symbol_returns) < 20 or len(symbol_returns) != len(benchmark_returns):
        return None

    symbol_mean = _mean(symbol_returns)
    benchmark_mean = _mean(benchmark_returns)
    covariance = _mean(
        [
            (s - symbol_mean) * (b - benchmark_mean)
            for s, b in zip(symbol_returns, benchmark_returns, strict=True)
        ]
    )
    symbol_variance = _mean([(s - symbol_mean) ** 2 for s in symbol_returns])
    benchmark_variance = _mean([(b - benchmark_mean) ** 2 for b in benchmark_returns])
    if benchmark_variance <= 0 or symbol_variance <= 0:
        return None

    beta = covariance / benchmark_variance
    correlation = covariance / math.sqrt(symbol_variance * benchmark_variance)

    if beta > 1.2:
        sensitivity = "amplifies market moves"
    elif beta < 0.8:
        sensitivity = "dampens market moves"
    else:
        sensitivity = "tracks the market roughly one for one"

    return MarketContextReading(
        name="beta_correlation",
        values={
            "beta": round(beta, 3),
            "correlation": round(correlation, 3),
            "window_bars": float(len(symbol_returns)),
        },
        direction=NEUTRAL,
        summary=(
            f"Beta {beta:.2f} and correlation {correlation:.2f} against {benchmark} over "
            f"{len(symbol_returns)} bars - {sensitivity}"
        ),
    )


def _benchmark_trend(
    benchmark_closes: Sequence[float], benchmark: str
) -> MarketContextReading | None:
    """The market's own trend: a tailwind or a headwind for everything in it."""
    if len(benchmark_closes) < BENCHMARK_TREND_WINDOW:
        return None
    average = _mean(benchmark_closes[-BENCHMARK_TREND_WINDOW:])
    latest = benchmark_closes[-1]
    above = latest > average
    return MarketContextReading(
        name="benchmark_trend",
        values={
            "benchmark_close": round(latest, 2),
            f"benchmark_sma_{BENCHMARK_TREND_WINDOW}": round(average, 2),
        },
        direction=BULLISH if above else BEARISH,
        summary=(
            f"{benchmark} at {latest:.2f} is {'above' if above else 'below'} its "
            f"{BENCHMARK_TREND_WINDOW}-bar average {average:.2f} - the market is a "
            f"{'tailwind' if above else 'headwind'}"
        ),
    )


def _drawdown_comparison(
    symbol_closes: Sequence[float], benchmark_closes: Sequence[float], benchmark: str
) -> MarketContextReading | None:
    """Who has held up better since their respective highs."""
    window = min(len(symbol_closes), RELATIVE_STRENGTH_WINDOWS["12m"])
    if window < STATS_WINDOW:
        return None

    def drawdown(closes: Sequence[float]) -> float:
        peak = max(closes[-window:])
        return (closes[-1] / peak - 1.0) * 100 if peak > 0 else 0.0

    symbol_drawdown = drawdown(symbol_closes)
    benchmark_drawdown = drawdown(benchmark_closes)
    spread = symbol_drawdown - benchmark_drawdown

    if spread > RELATIVE_STRENGTH_THRESHOLD_PCT:
        direction, note = BULLISH, "holding up better than the market"
    elif spread < -RELATIVE_STRENGTH_THRESHOLD_PCT:
        direction, note = BEARISH, "falling further than the market"
    else:
        direction, note = NEUTRAL, "drawing down in line with the market"

    return MarketContextReading(
        name="drawdown_vs_benchmark",
        values={
            "symbol_drawdown_pct": round(symbol_drawdown, 2),
            "benchmark_drawdown_pct": round(benchmark_drawdown, 2),
            "spread_pct": round(spread, 2),
        },
        direction=direction,
        summary=(
            f"Symbol is {symbol_drawdown:.1f}% off its {window}-bar high against "
            f"{benchmark}'s {benchmark_drawdown:.1f}% - {note}"
        ),
    )
