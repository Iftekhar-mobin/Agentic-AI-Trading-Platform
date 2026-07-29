"""Tests for the deterministic market-context (benchmark comparison) service."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from atp.domain.errors import InsufficientHistoryError
from atp.domain.models.analysis import SignalDirection
from atp.domain.models.market import Bar, BarInterval, PriceHistory
from atp.domain.services.market_context import align, analyze

START = datetime(2026, 1, 1, tzinfo=UTC)


def history(
    closes: list[float], *, symbol: str = "AAPL", skip: set[int] | None = None
) -> PriceHistory:
    """Build a series; ``skip`` drops bars to simulate a halt or holiday."""
    skipped = skip or set()
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
        if index not in skipped
    )
    return PriceHistory(symbol=symbol, interval=BarInterval.DAY_1, bars=bars)


def growth(start: float, rate: float, count: int = 300) -> list[float]:
    return [start * (1 + rate) ** index for index in range(count)]


def readings_by_name(symbol: PriceHistory, benchmark: PriceHistory) -> dict[str, object]:
    return {reading.name: reading for reading in analyze(symbol, benchmark)}


# --- Alignment --------------------------------------------------------------


def test_alignment_keeps_only_shared_timestamps() -> None:
    """A halted week in one series must not shift the other's returns."""
    symbol = history(growth(100, 0.001), skip={10, 11, 12})
    benchmark = history(growth(400, 0.001), symbol="SPY")

    symbol_closes, benchmark_closes = align(symbol, benchmark)

    assert len(symbol_closes) == len(benchmark_closes) == len(symbol)
    assert len(symbol_closes) == 297


def test_no_overlap_is_rejected() -> None:
    symbol = history(growth(100, 0.001, 100))
    benchmark = PriceHistory(
        symbol="SPY",
        interval=BarInterval.DAY_1,
        bars=tuple(
            Bar(
                timestamp=START + timedelta(days=1000 + index),
                open=400.0,
                high=404.0,
                low=396.0,
                close=400.0,
                volume=1_000.0,
            )
            for index in range(100)
        ),
    )

    with pytest.raises(InsufficientHistoryError, match="overlapping bars"):
        analyze(symbol, benchmark)


def test_short_overlap_is_rejected() -> None:
    symbol = history(growth(100, 0.001, 40))
    benchmark = history(growth(400, 0.001, 40), symbol="SPY")

    with pytest.raises(InsufficientHistoryError, match="required for market research"):
        analyze(symbol, benchmark)


# --- Relative strength ------------------------------------------------------


def test_outperformance_reads_bullish() -> None:
    symbol = history(growth(100, 0.004))
    benchmark = history(growth(400, 0.001), symbol="SPY")

    reading = readings_by_name(symbol, benchmark)["relative_strength_3m"]
    assert reading.direction is SignalDirection.BULLISH  # type: ignore[attr-defined]
    assert reading.values["spread_pct"] > 0  # type: ignore[attr-defined]
    assert "outperforming" in reading.summary  # type: ignore[attr-defined]


def test_underperformance_reads_bearish() -> None:
    symbol = history(growth(100, 0.0))
    benchmark = history(growth(400, 0.004), symbol="SPY")

    reading = readings_by_name(symbol, benchmark)["relative_strength_3m"]
    assert reading.direction is SignalDirection.BEARISH  # type: ignore[attr-defined]
    assert reading.values["spread_pct"] < 0  # type: ignore[attr-defined]


def test_moving_with_the_market_reads_neutral() -> None:
    """Identical paths at different price levels are the definition of no signal."""
    symbol = history(growth(100, 0.002))
    benchmark = history(growth(400, 0.002), symbol="SPY")

    reading = readings_by_name(symbol, benchmark)["relative_strength_3m"]
    assert reading.direction is SignalDirection.NEUTRAL  # type: ignore[attr-defined]
    assert reading.values["spread_pct"] == pytest.approx(0.0, abs=0.01)  # type: ignore[attr-defined]


def test_windows_that_do_not_fit_are_omitted() -> None:
    """A 12-month reading needs 12 months of overlap; it is skipped, not faked."""
    symbol = history(growth(100, 0.002, 90))
    benchmark = history(growth(400, 0.002, 90), symbol="SPY")

    names = set(readings_by_name(symbol, benchmark))
    assert "relative_strength_1m" in names
    assert "relative_strength_3m" in names
    assert "relative_strength_12m" not in names


# --- Beta and correlation ---------------------------------------------------


def test_a_symbol_that_doubles_every_market_move_has_beta_two() -> None:
    benchmark_returns = [0.01 if index % 2 else -0.005 for index in range(300)]
    benchmark_closes = [400.0]
    symbol_closes = [100.0]
    for change in benchmark_returns:
        benchmark_closes.append(benchmark_closes[-1] * (1 + change))
        symbol_closes.append(symbol_closes[-1] * (1 + 2 * change))

    reading = readings_by_name(history(symbol_closes), history(benchmark_closes, symbol="SPY"))[
        "beta_correlation"
    ]

    assert reading.values["beta"] == pytest.approx(2.0, abs=0.1)  # type: ignore[attr-defined]
    assert reading.values["correlation"] == pytest.approx(1.0, abs=0.01)  # type: ignore[attr-defined]
    assert "amplifies" in reading.summary  # type: ignore[attr-defined]


def test_beta_and_correlation_are_context_not_a_direction() -> None:
    symbol = history(growth(100, 0.002))
    benchmark = history(growth(400, 0.002), symbol="SPY")

    reading = readings_by_name(symbol, benchmark)["beta_correlation"]
    assert reading.direction is SignalDirection.NEUTRAL  # type: ignore[attr-defined]


# --- Benchmark trend --------------------------------------------------------


def test_a_rising_benchmark_is_a_tailwind() -> None:
    symbol = history(growth(100, 0.002))
    benchmark = history(growth(400, 0.003), symbol="SPY")

    reading = readings_by_name(symbol, benchmark)["benchmark_trend"]
    assert reading.direction is SignalDirection.BULLISH  # type: ignore[attr-defined]
    assert "tailwind" in reading.summary  # type: ignore[attr-defined]


def test_a_falling_benchmark_is_a_headwind() -> None:
    symbol = history(growth(100, 0.002))
    benchmark = history(growth(400, -0.003), symbol="SPY")

    reading = readings_by_name(symbol, benchmark)["benchmark_trend"]
    assert reading.direction is SignalDirection.BEARISH  # type: ignore[attr-defined]
    assert "headwind" in reading.summary  # type: ignore[attr-defined]


# --- Drawdown ---------------------------------------------------------------


def test_holding_up_better_than_the_market_reads_bullish() -> None:
    """Both sell off; the symbol falls less."""
    symbol_closes = [
        *growth(100, 0.002, 200),
        *[100 * 1.002**199 * (1 - 0.0005 * i) for i in range(100)],
    ]
    benchmark_closes = [
        *growth(400, 0.002, 200),
        *[400 * 1.002**199 * (1 - 0.003 * i) for i in range(100)],
    ]

    reading = readings_by_name(history(symbol_closes), history(benchmark_closes, symbol="SPY"))[
        "drawdown_vs_benchmark"
    ]

    assert reading.direction is SignalDirection.BULLISH  # type: ignore[attr-defined]
    assert reading.values["spread_pct"] > 0  # type: ignore[attr-defined]


def test_every_reading_has_a_summary_carrying_its_numbers() -> None:
    symbol = history(growth(100, 0.003))
    benchmark = history(growth(400, 0.001), symbol="SPY")

    readings = analyze(symbol, benchmark)
    assert len(readings) >= 5
    assert all(reading.summary for reading in readings)
    assert all(reading.values for reading in readings)


def test_analysis_is_reproducible() -> None:
    symbol = history(growth(100, 0.003))
    benchmark = history(growth(400, 0.001), symbol="SPY")

    assert analyze(symbol, benchmark) == analyze(symbol, benchmark)
