"""Golden-value tests for deterministic chart pattern detection.

Each test builds a price path with the shape it is testing for, so a failure
means the detector changed behaviour rather than that the market moved.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from atp.domain.errors import InsufficientHistoryError
from atp.domain.models.analysis import SignalDirection
from atp.domain.models.market import Bar, BarInterval, PriceHistory
from atp.domain.models.patterns import ChartPattern, LevelKind, PatternKind
from atp.domain.services.chart_patterns import (
    detect_patterns,
    find_levels,
    find_swings,
    net_direction,
    scan,
)

START = datetime(2026, 1, 1, tzinfo=UTC)


def history(closes: list[float]) -> PriceHistory:
    bars = tuple(
        Bar(
            timestamp=START + timedelta(days=index),
            open=close,
            high=close * 1.005,
            low=close * 0.995,
            close=close,
            volume=1_000.0,
        )
        for index, close in enumerate(closes)
    )
    return PriceHistory(symbol="AAPL", interval=BarInterval.DAY_1, bars=bars)


def legs(start: float, *segments: tuple[float, int]) -> list[float]:
    """Build a zig-zag price path from (target, steps) legs.

    Each leg excludes its starting value, so joining two legs leaves a single
    bar at the turn rather than a two-bar plateau. That matters: swing
    detection requires a strict local extreme, and a duplicated endpoint would
    hide every peak this file is trying to test.
    """
    closes = [start]
    current = start
    for target, steps in segments:
        step = (target - current) / steps
        closes.extend(current + step * index for index in range(1, steps + 1))
        current = target
    return closes


def flat(value: float, steps: int) -> list[float]:
    return [value] * steps


def kinds(patterns: tuple[ChartPattern, ...]) -> set[PatternKind]:
    return {pattern.kind for pattern in patterns}


def find(patterns: tuple[ChartPattern, ...], kind: PatternKind) -> ChartPattern:
    match = next((pattern for pattern in patterns if pattern.kind is kind), None)
    assert match is not None, f"{kind.value} not detected; got {[p.kind.value for p in patterns]}"
    return match


def patterns_for(closes: list[float]) -> tuple[ChartPattern, ...]:
    result = history(closes)
    return detect_patterns(result, find_swings(result))


# --- Swing points -----------------------------------------------------------


def test_swings_are_local_extremes() -> None:
    closes = legs(100, (120, 10), (100, 10), (118, 10))
    swings = find_swings(history(closes))

    assert swings
    assert {swing.kind.value for swing in swings} == {"high", "low"}
    # Every swing sits strictly inside the series, never on the padding edges.
    assert all(2 < swing.index < len(closes) - 3 for swing in swings)


def test_a_flat_series_has_no_swings() -> None:
    """Equal neighbours must not each qualify as an extreme."""
    assert find_swings(history(flat(100.0, 40))) == ()


# --- Levels -----------------------------------------------------------------


def test_levels_need_at_least_two_touches() -> None:
    closes = legs(100, (120, 8), (105, 8), (120, 8), (106, 8), (118, 8))
    result = history(closes)
    levels = find_levels(find_swings(result), result.bars[-1].close)

    assert levels
    assert all(level.touches >= 2 for level in levels)


def test_levels_are_classified_against_the_latest_close() -> None:
    closes = legs(100, (120, 8), (100, 8), (120, 8), (100, 8), (110, 8))
    result = history(closes)
    latest = result.bars[-1].close
    levels = find_levels(find_swings(result), latest)

    for level in levels:
        expected = LevelKind.RESISTANCE if level.price > latest else LevelKind.SUPPORT
        assert level.kind is expected
    assert levels == tuple(sorted(levels, key=lambda lv: abs(lv.distance_pct)))


def test_no_swings_means_no_levels() -> None:
    assert find_levels((), 100.0) == ()


# --- Double top / bottom ----------------------------------------------------


def test_double_top_is_detected_and_unconfirmed_while_above_the_neckline() -> None:
    closes = legs(100, (130, 10), (118, 8), (129, 8), (124, 6))
    pattern = find(patterns_for(closes), PatternKind.DOUBLE_TOP)

    assert pattern.direction is SignalDirection.BEARISH
    assert pattern.confirmed is False
    assert "unconfirmed" in pattern.summary
    assert pattern.levels["neckline"] == pytest.approx(117.4, abs=1.0)


def test_double_top_confirms_once_the_neckline_breaks() -> None:
    closes = legs(100, (130, 10), (118, 8), (129, 8), (110, 8))
    pattern = find(patterns_for(closes), PatternKind.DOUBLE_TOP)

    assert pattern.confirmed is True
    assert "confirmed" in pattern.summary


def test_double_bottom_is_detected() -> None:
    closes = legs(130, (100, 10), (112, 8), (101, 8), (106, 6))
    pattern = find(patterns_for(closes), PatternKind.DOUBLE_BOTTOM)

    assert pattern.direction is SignalDirection.BULLISH
    assert pattern.confirmed is False


def test_a_shallow_pullback_is_not_a_double_top() -> None:
    """Two peaks with no real trough between them is noise, not a formation."""
    closes = legs(100, (130, 10), (129, 6), (130, 6), (128, 6))
    assert PatternKind.DOUBLE_TOP not in kinds(patterns_for(closes))


def test_unequal_peaks_are_not_a_double_top() -> None:
    closes = legs(100, (130, 10), (115, 8), (150, 8), (140, 6))
    assert PatternKind.DOUBLE_TOP not in kinds(patterns_for(closes))


# --- Head and shoulders -----------------------------------------------------


def test_head_and_shoulders_is_detected() -> None:
    closes = legs(100, (120, 8), (108, 6), (140, 8), (109, 6), (121, 8), (115, 6))
    pattern = find(patterns_for(closes), PatternKind.HEAD_AND_SHOULDERS)

    assert pattern.direction is SignalDirection.BEARISH
    assert pattern.levels["head"] > pattern.levels["left_shoulder"]
    assert pattern.levels["head"] > pattern.levels["right_shoulder"]


def test_inverse_head_and_shoulders_is_detected() -> None:
    closes = legs(140, (120, 8), (132, 6), (100, 8), (131, 6), (119, 8), (126, 6))
    pattern = find(patterns_for(closes), PatternKind.INVERSE_HEAD_AND_SHOULDERS)

    assert pattern.direction is SignalDirection.BULLISH
    assert pattern.levels["head"] < pattern.levels["left_shoulder"]


def test_a_lower_middle_peak_is_not_head_and_shoulders() -> None:
    closes = legs(100, (130, 8), (115, 6), (120, 8), (114, 6), (131, 8), (125, 6))
    assert PatternKind.HEAD_AND_SHOULDERS not in kinds(patterns_for(closes))


# --- Trend structure --------------------------------------------------------


def test_higher_highs_and_higher_lows_read_as_an_uptrend() -> None:
    closes = legs(100, (115, 8), (105, 6), (125, 8), (112, 6), (130, 8))
    pattern = find(patterns_for(closes), PatternKind.UPTREND_STRUCTURE)

    assert pattern.direction is SignalDirection.BULLISH
    assert pattern.confirmed is True
    assert pattern.levels["recent_high"] > pattern.levels["prior_high"]


def test_lower_highs_and_lower_lows_read_as_a_downtrend() -> None:
    closes = legs(130, (115, 8), (125, 6), (105, 8), (118, 6), (100, 8))
    pattern = find(patterns_for(closes), PatternKind.DOWNTREND_STRUCTURE)

    assert pattern.direction is SignalDirection.BEARISH


# --- Range break ------------------------------------------------------------


def test_a_close_above_the_range_is_a_breakout() -> None:
    closes = [*flat(100.0, 20), *legs(100, (104, 20)), 130.0]
    pattern = find(patterns_for(closes), PatternKind.RANGE_BREAKOUT)

    assert pattern.direction is SignalDirection.BULLISH
    assert pattern.confirmed is True
    assert pattern.levels["close"] > pattern.levels["range_high"]


def test_a_close_below_the_range_is_a_breakdown() -> None:
    closes = [*flat(100.0, 20), *legs(100, (97, 20)), 70.0]
    pattern = find(patterns_for(closes), PatternKind.RANGE_BREAKDOWN)

    assert pattern.direction is SignalDirection.BEARISH
    assert pattern.levels["close"] < pattern.levels["range_low"]


def test_price_inside_the_range_is_not_a_break() -> None:
    closes = legs(100, (110, 20), (100, 20))
    detected = kinds(patterns_for(closes))
    assert PatternKind.RANGE_BREAKOUT not in detected
    assert PatternKind.RANGE_BREAKDOWN not in detected


# --- Net direction ----------------------------------------------------------


def make_pattern(
    direction: SignalDirection, *, confirmed: bool, quality: float = 1.0
) -> ChartPattern:
    return ChartPattern(
        kind=PatternKind.RANGE_BREAKOUT,
        direction=direction,
        start=START,
        end=START + timedelta(days=1),
        levels={},
        confirmed=confirmed,
        quality=quality,
        summary="x",
    )


def test_confirmed_patterns_outweigh_forming_ones() -> None:
    """One confirmed bearish beats one forming bullish of the same quality."""
    result = net_direction(
        (
            make_pattern(SignalDirection.BEARISH, confirmed=True),
            make_pattern(SignalDirection.BULLISH, confirmed=False),
        )
    )
    assert result is SignalDirection.BEARISH


def test_opposing_confirmed_patterns_cancel_to_neutral() -> None:
    result = net_direction(
        (
            make_pattern(SignalDirection.BEARISH, confirmed=True),
            make_pattern(SignalDirection.BULLISH, confirmed=True),
        )
    )
    assert result is SignalDirection.NEUTRAL


def test_low_quality_patterns_barely_move_the_needle() -> None:
    result = net_direction((make_pattern(SignalDirection.BULLISH, confirmed=True, quality=0.1),))
    assert result is SignalDirection.NEUTRAL


def test_no_patterns_is_neutral() -> None:
    assert net_direction(()) is SignalDirection.NEUTRAL


# --- Scan -------------------------------------------------------------------


def test_scan_returns_swings_levels_and_patterns() -> None:
    closes = legs(100, (130, 20), (115, 20), (132, 25))
    swings, levels, patterns = scan(history(closes))

    assert swings
    assert isinstance(levels, tuple)
    assert isinstance(patterns, tuple)


def test_scan_rejects_short_history() -> None:
    with pytest.raises(InsufficientHistoryError, match="required to detect chart patterns"):
        scan(history(legs(100, (110, 30))))


def test_scan_is_reproducible() -> None:
    closes = legs(100, (130, 20), (115, 20), (132, 25))
    assert scan(history(closes)) == scan(history(closes))
