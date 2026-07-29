"""Deterministic chart pattern detection.

Pure geometry over swing points, in the domain and in plain stdlib — no pandas,
no port. It sits alongside ``regime_detection`` rather than behind an interface
like the indicator engine because it needs nothing from the outside world, and
the same bars must always produce the same shapes.

The detectors are deliberately conservative. Every one requires explicit
proportions (how close two peaks must be, how deep the trough between them
must run) rather than a loose "looks like a top", and every result carries a
``quality`` score derived from how well the shape actually fits its ideal. The
failure mode this guards against is the classic one: given enough latitude,
something is always a head and shoulders.

``confirmed`` separates a completed pattern from a forming one. A double top
whose neckline has not broken is a warning; one that has broken is a signal,
and conflating the two is how pattern trading loses money.
"""

from __future__ import annotations

from datetime import datetime

from atp.domain.errors import InsufficientHistoryError
from atp.domain.models.analysis import SignalDirection
from atp.domain.models.market import Bar, PriceHistory
from atp.domain.models.patterns import (
    MIN_PATTERN_BARS,
    ChartPattern,
    LevelKind,
    PatternKind,
    PriceLevel,
    SwingKind,
    SwingPoint,
)

BULLISH = SignalDirection.BULLISH
BEARISH = SignalDirection.BEARISH
NEUTRAL = SignalDirection.NEUTRAL

SWING_WINDOW = 3
"""Bars either side that a swing point must dominate to count as an extreme."""

PEAK_TOLERANCE_PCT = 3.0
"""How close two peaks must be to read as 'equal' (double top, H&S shoulders)."""

MIN_TROUGH_DEPTH_PCT = 3.0
"""The pullback between two peaks must be at least this deep, or it is noise."""

LEVEL_TOLERANCE_PCT = 1.5
"""Swings within this distance of each other cluster into one price level."""

RANGE_LOOKBACK = 40
"""Bars used to define the recent trading range for breakout detection."""

MAX_LEVELS = 6
"""Report only the nearest levels; a chart with thirty 'key levels' has none."""


def _pct_diff(left: float, right: float) -> float:
    """Absolute difference as a percentage of the average of the two."""
    midpoint = (left + right) / 2
    return abs(left - right) / midpoint * 100 if midpoint > 0 else 100.0


def _slope(points: list[tuple[int, float]]) -> float:
    """Least-squares slope in price per bar."""
    count = len(points)
    if count < 2:
        return 0.0
    mean_x = sum(x for x, _ in points) / count
    mean_y = sum(y for _, y in points) / count
    denominator = sum((x - mean_x) ** 2 for x, _ in points)
    if denominator == 0:
        return 0.0
    return sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator


def find_swings(history: PriceHistory, *, window: int = SWING_WINDOW) -> tuple[SwingPoint, ...]:
    """Fractal swing points: a bar whose high (low) dominates its neighbours.

    A bar equal to its neighbours does not qualify — flat stretches would
    otherwise produce a swing on every bar.
    """
    bars = history.bars
    swings: list[SwingPoint] = []
    for index in range(window, len(bars) - window):
        bar = bars[index]
        neighbours = [*bars[index - window : index], *bars[index + 1 : index + window + 1]]
        if all(bar.high > other.high for other in neighbours):
            swings.append(
                SwingPoint(
                    kind=SwingKind.HIGH,
                    index=index,
                    timestamp=bar.timestamp,
                    price=bar.high,
                )
            )
        elif all(bar.low < other.low for other in neighbours):
            swings.append(
                SwingPoint(
                    kind=SwingKind.LOW,
                    index=index,
                    timestamp=bar.timestamp,
                    price=bar.low,
                )
            )
    return tuple(swings)


def find_levels(
    swings: tuple[SwingPoint, ...],
    latest_close: float,
    *,
    tolerance_pct: float = LEVEL_TOLERANCE_PCT,
    max_levels: int = MAX_LEVELS,
) -> tuple[PriceLevel, ...]:
    """Cluster swing prices into support and resistance levels.

    A level needs at least two touches: one swing is a turning point, two is a
    level the market has actually respected.
    """
    if not swings:
        return ()

    clusters: list[list[SwingPoint]] = []
    for swing in sorted(swings, key=lambda point: point.price):
        if clusters and _pct_diff(clusters[-1][-1].price, swing.price) <= tolerance_pct:
            clusters[-1].append(swing)
        else:
            clusters.append([swing])

    levels = [
        PriceLevel(
            kind=LevelKind.RESISTANCE if price > latest_close else LevelKind.SUPPORT,
            price=round(price, 4),
            touches=len(cluster),
            last_touch=max(point.timestamp for point in cluster),
            distance_pct=round((price - latest_close) / latest_close * 100, 2),
        )
        for cluster in clusters
        if len(cluster) >= 2
        for price in [sum(point.price for point in cluster) / len(cluster)]
    ]
    # Nearest to price first: those are the ones that matter to a decision now.
    levels.sort(key=lambda level: abs(level.distance_pct))
    return tuple(levels[:max_levels])


def detect_patterns(
    history: PriceHistory, swings: tuple[SwingPoint, ...]
) -> tuple[ChartPattern, ...]:
    """Run every detector; return the shapes that actually fit."""
    bars = history.bars
    latest = bars[-1]
    highs = [swing for swing in swings if swing.kind is SwingKind.HIGH]
    lows = [swing for swing in swings if swing.kind is SwingKind.LOW]

    candidates = [
        _double_top(highs, lows, latest),
        _double_bottom(highs, lows, latest),
        _head_and_shoulders(highs, lows, latest),
        _inverse_head_and_shoulders(highs, lows, latest),
        _trend_structure(highs, lows, latest),
        _triangle(highs, lows, latest),
        _range_break(bars),
    ]
    return tuple(pattern for pattern in candidates if pattern is not None)


def _double_top(
    highs: list[SwingPoint], lows: list[SwingPoint], latest: Bar
) -> ChartPattern | None:
    if len(highs) < 2:
        return None
    first, second = highs[-2], highs[-1]
    if _pct_diff(first.price, second.price) > PEAK_TOLERANCE_PCT:
        return None

    between = [low for low in lows if first.index < low.index < second.index]
    if not between:
        return None
    trough = min(between, key=lambda low: low.price)
    peak = max(first.price, second.price)
    depth_pct = (peak - trough.price) / peak * 100
    if depth_pct < MIN_TROUGH_DEPTH_PCT:
        return None

    confirmed = latest.close < trough.price
    symmetry = 1.0 - _pct_diff(first.price, second.price) / PEAK_TOLERANCE_PCT
    return ChartPattern(
        kind=PatternKind.DOUBLE_TOP,
        direction=BEARISH,
        start=first.timestamp,
        end=second.timestamp,
        levels={
            "peak_1": round(first.price, 4),
            "peak_2": round(second.price, 4),
            "neckline": round(trough.price, 4),
        },
        confirmed=confirmed,
        quality=round(max(0.0, min(1.0, symmetry)), 3),
        summary=(
            f"Double top at {first.price:.2f}/{second.price:.2f} with a neckline at "
            f"{trough.price:.2f} - "
            + (
                f"confirmed, price closed below the neckline at {latest.close:.2f}"
                if confirmed
                else f"unconfirmed, price is still above the neckline at {latest.close:.2f}"
            )
        ),
    )


def _double_bottom(
    highs: list[SwingPoint], lows: list[SwingPoint], latest: Bar
) -> ChartPattern | None:
    if len(lows) < 2:
        return None
    first, second = lows[-2], lows[-1]
    if _pct_diff(first.price, second.price) > PEAK_TOLERANCE_PCT:
        return None

    between = [high for high in highs if first.index < high.index < second.index]
    if not between:
        return None
    peak = max(between, key=lambda high: high.price)
    trough = min(first.price, second.price)
    height_pct = (peak.price - trough) / trough * 100
    if height_pct < MIN_TROUGH_DEPTH_PCT:
        return None

    confirmed = latest.close > peak.price
    symmetry = 1.0 - _pct_diff(first.price, second.price) / PEAK_TOLERANCE_PCT
    return ChartPattern(
        kind=PatternKind.DOUBLE_BOTTOM,
        direction=BULLISH,
        start=first.timestamp,
        end=second.timestamp,
        levels={
            "trough_1": round(first.price, 4),
            "trough_2": round(second.price, 4),
            "neckline": round(peak.price, 4),
        },
        confirmed=confirmed,
        quality=round(max(0.0, min(1.0, symmetry)), 3),
        summary=(
            f"Double bottom at {first.price:.2f}/{second.price:.2f} with a neckline at "
            f"{peak.price:.2f} - "
            + (
                f"confirmed, price closed above the neckline at {latest.close:.2f}"
                if confirmed
                else f"unconfirmed, price is still below the neckline at {latest.close:.2f}"
            )
        ),
    )


def _head_and_shoulders(
    highs: list[SwingPoint], lows: list[SwingPoint], latest: Bar
) -> ChartPattern | None:
    if len(highs) < 3:
        return None
    left, head, right = highs[-3], highs[-2], highs[-1]
    if not (head.price > left.price and head.price > right.price):
        return None
    if _pct_diff(left.price, right.price) > PEAK_TOLERANCE_PCT:
        return None

    necklines = [low.price for low in lows if left.index < low.index < right.index]
    if len(necklines) < 2:
        return None
    neckline = sum(necklines) / len(necklines)

    confirmed = latest.close < neckline
    symmetry = 1.0 - _pct_diff(left.price, right.price) / PEAK_TOLERANCE_PCT
    return ChartPattern(
        kind=PatternKind.HEAD_AND_SHOULDERS,
        direction=BEARISH,
        start=left.timestamp,
        end=right.timestamp,
        levels={
            "left_shoulder": round(left.price, 4),
            "head": round(head.price, 4),
            "right_shoulder": round(right.price, 4),
            "neckline": round(neckline, 4),
        },
        confirmed=confirmed,
        quality=round(max(0.0, min(1.0, symmetry)), 3),
        summary=(
            f"Head and shoulders: head {head.price:.2f} above shoulders "
            f"{left.price:.2f}/{right.price:.2f}, neckline {neckline:.2f} - "
            + ("neckline broken" if confirmed else "neckline intact")
        ),
    )


def _inverse_head_and_shoulders(
    highs: list[SwingPoint], lows: list[SwingPoint], latest: Bar
) -> ChartPattern | None:
    if len(lows) < 3:
        return None
    left, head, right = lows[-3], lows[-2], lows[-1]
    if not (head.price < left.price and head.price < right.price):
        return None
    if _pct_diff(left.price, right.price) > PEAK_TOLERANCE_PCT:
        return None

    necklines = [high.price for high in highs if left.index < high.index < right.index]
    if len(necklines) < 2:
        return None
    neckline = sum(necklines) / len(necklines)

    confirmed = latest.close > neckline
    symmetry = 1.0 - _pct_diff(left.price, right.price) / PEAK_TOLERANCE_PCT
    return ChartPattern(
        kind=PatternKind.INVERSE_HEAD_AND_SHOULDERS,
        direction=BULLISH,
        start=left.timestamp,
        end=right.timestamp,
        levels={
            "left_shoulder": round(left.price, 4),
            "head": round(head.price, 4),
            "right_shoulder": round(right.price, 4),
            "neckline": round(neckline, 4),
        },
        confirmed=confirmed,
        quality=round(max(0.0, min(1.0, symmetry)), 3),
        summary=(
            f"Inverse head and shoulders: head {head.price:.2f} below shoulders "
            f"{left.price:.2f}/{right.price:.2f}, neckline {neckline:.2f} - "
            + ("neckline broken" if confirmed else "neckline intact")
        ),
    )


def _trend_structure(
    highs: list[SwingPoint], lows: list[SwingPoint], latest: Bar
) -> ChartPattern | None:
    """Higher highs with higher lows, or the mirror — the definition of a trend."""
    if len(highs) < 2 or len(lows) < 2:
        return None
    rising = highs[-1].price > highs[-2].price and lows[-1].price > lows[-2].price
    falling = highs[-1].price < highs[-2].price and lows[-1].price < lows[-2].price
    if not (rising or falling):
        return None

    start = min(highs[-2].timestamp, lows[-2].timestamp)
    levels = {
        "prior_high": round(highs[-2].price, 4),
        "recent_high": round(highs[-1].price, 4),
        "prior_low": round(lows[-2].price, 4),
        "recent_low": round(lows[-1].price, 4),
    }
    if rising:
        return ChartPattern(
            kind=PatternKind.UPTREND_STRUCTURE,
            direction=BULLISH,
            start=start,
            end=latest.timestamp,
            levels=levels,
            confirmed=True,  # structure is observed, not awaiting a break
            quality=0.6,
            summary=(
                f"Uptrend structure: higher high {highs[-1].price:.2f} over "
                f"{highs[-2].price:.2f}, higher low {lows[-1].price:.2f} over "
                f"{lows[-2].price:.2f}"
            ),
        )
    return ChartPattern(
        kind=PatternKind.DOWNTREND_STRUCTURE,
        direction=BEARISH,
        start=start,
        end=latest.timestamp,
        levels=levels,
        confirmed=True,
        quality=0.6,
        summary=(
            f"Downtrend structure: lower high {highs[-1].price:.2f} under "
            f"{highs[-2].price:.2f}, lower low {lows[-1].price:.2f} under "
            f"{lows[-2].price:.2f}"
        ),
    )


def _triangle(highs: list[SwingPoint], lows: list[SwingPoint], latest: Bar) -> ChartPattern | None:
    """Converging trendlines through the last three highs and lows."""
    if len(highs) < 3 or len(lows) < 3:
        return None

    recent_highs = highs[-3:]
    recent_lows = lows[-3:]
    high_slope = _slope([(point.index, point.price) for point in recent_highs])
    low_slope = _slope([(point.index, point.price) for point in recent_lows])

    # Express slopes as percent of price per bar so thresholds are scale-free.
    scale = latest.close / 100 if latest.close > 0 else 1.0
    high_pct = high_slope / scale
    low_pct = low_slope / scale
    flat = 0.05  # percent per bar

    start = min(recent_highs[0].timestamp, recent_lows[0].timestamp)
    levels = {
        "resistance": round(recent_highs[-1].price, 4),
        "support": round(recent_lows[-1].price, 4),
        "high_slope_pct_per_bar": round(high_pct, 4),
        "low_slope_pct_per_bar": round(low_pct, 4),
    }

    if abs(high_pct) <= flat and low_pct > flat:
        kind, direction, shape = (
            PatternKind.ASCENDING_TRIANGLE,
            BULLISH,
            ("flat resistance with rising lows"),
        )
    elif abs(low_pct) <= flat and high_pct < -flat:
        kind, direction, shape = (
            PatternKind.DESCENDING_TRIANGLE,
            BEARISH,
            ("flat support with falling highs"),
        )
    elif high_pct < -flat and low_pct > flat:
        kind, direction, shape = (
            PatternKind.SYMMETRICAL_TRIANGLE,
            NEUTRAL,
            ("falling highs and rising lows"),
        )
    else:
        return None

    return ChartPattern(
        kind=kind,
        direction=direction,
        start=start,
        end=latest.timestamp,
        levels=levels,
        confirmed=False,  # a triangle resolves on the break, which has not happened
        quality=0.5,
        summary=(
            f"{kind.value.replace('_', ' ').capitalize()}: {shape}, resistance near "
            f"{recent_highs[-1].price:.2f} and support near {recent_lows[-1].price:.2f}"
        ),
    )


def _range_break(bars: tuple[Bar, ...]) -> ChartPattern | None:
    """A close outside the prior range — the simplest confirmed pattern there is."""
    if len(bars) < RANGE_LOOKBACK + 1:
        return None
    window = bars[-(RANGE_LOOKBACK + 1) : -1]
    latest = bars[-1]
    range_high = max(bar.high for bar in window)
    range_low = min(bar.low for bar in window)

    if latest.close > range_high:
        kind, direction = PatternKind.RANGE_BREAKOUT, BULLISH
        note = f"closed above the {RANGE_LOOKBACK}-bar high {range_high:.2f}"
    elif latest.close < range_low:
        kind, direction = PatternKind.RANGE_BREAKDOWN, BEARISH
        note = f"closed below the {RANGE_LOOKBACK}-bar low {range_low:.2f}"
    else:
        return None

    span_pct = (range_high - range_low) / range_low * 100 if range_low > 0 else 0.0
    return ChartPattern(
        kind=kind,
        direction=direction,
        start=window[0].timestamp,
        end=latest.timestamp,
        levels={
            "range_high": round(range_high, 4),
            "range_low": round(range_low, 4),
            "close": round(latest.close, 4),
        },
        confirmed=True,
        # A break out of a tight range is a weaker signal than out of a wide one.
        quality=round(min(1.0, span_pct / 20), 3),
        summary=f"Range break: price {note} at {latest.close:.2f}",
    )


def net_direction(patterns: tuple[ChartPattern, ...]) -> SignalDirection:
    """Weigh confirmed patterns by quality; unconfirmed shapes count for half.

    A forming double top is information, but it is not the same evidence as one
    whose neckline has broken, and the arithmetic should say so.
    """
    score = 0.0
    for pattern in patterns:
        if pattern.direction is NEUTRAL:
            continue
        weight = pattern.quality * (1.0 if pattern.confirmed else 0.5)
        score += weight if pattern.direction is BULLISH else -weight
    if score > 0.25:
        return BULLISH
    if score < -0.25:
        return BEARISH
    return NEUTRAL


def scan(
    history: PriceHistory,
) -> tuple[tuple[SwingPoint, ...], tuple[PriceLevel, ...], tuple[ChartPattern, ...]]:
    """Full deterministic scan of one timeframe: swings, levels, patterns."""
    if len(history) < MIN_PATTERN_BARS:
        msg = (
            f"{history.symbol}/{history.interval.value}: {len(history)} bars available, "
            f"{MIN_PATTERN_BARS} required to detect chart patterns"
        )
        raise InsufficientHistoryError(msg)

    latest_close = history.bars[-1].close
    swings = find_swings(history)
    levels = find_levels(swings, latest_close)
    patterns = detect_patterns(history, swings)
    return swings, levels, patterns


def latest_timestamp(history: PriceHistory) -> datetime:
    return history.bars[-1].timestamp
