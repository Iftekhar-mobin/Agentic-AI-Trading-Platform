"""Golden-value tests for the deterministic fundamental classification rules."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from atp.domain.models.analysis import SignalDirection
from atp.domain.models.fundamentals import FundamentalCategory, Fundamentals
from atp.domain.services.fundamental_rules import classify

AS_OF = datetime(2026, 7, 22, tzinfo=UTC)

BULLISH = SignalDirection.BULLISH
BEARISH = SignalDirection.BEARISH
NEUTRAL = SignalDirection.NEUTRAL


def snapshot(**metrics: float | str | None) -> Fundamentals:
    return Fundamentals(symbol="AAPL", as_of=AS_OF, **metrics)


def direction_of(name: str, **metrics: float | str | None) -> SignalDirection:
    readings = {reading.name: reading for reading in classify(snapshot(**metrics))}
    return readings[name].direction


@pytest.mark.parametrize(
    ("value", "expected"),
    [(9.0, BULLISH), (22.0, NEUTRAL), (44.0, BEARISH)],
)
def test_trailing_pe_bands(value: float, expected: SignalDirection) -> None:
    assert direction_of("trailing_pe", trailing_pe=value) is expected


def test_negative_trailing_pe_is_bearish_not_cheap() -> None:
    readings = {reading.name: reading for reading in classify(snapshot(trailing_pe=-12.0))}
    reading = readings["trailing_pe"]
    assert reading.direction is BEARISH
    assert "not profitable" in reading.summary


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.6, BULLISH), (1.5, NEUTRAL), (3.0, BEARISH)],
)
def test_peg_bands(value: float, expected: SignalDirection) -> None:
    assert direction_of("peg_ratio", peg_ratio=value) is expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.30, BULLISH), (0.09, NEUTRAL), (-0.04, BEARISH)],
)
def test_profit_margin_bands(value: float, expected: SignalDirection) -> None:
    assert direction_of("profit_margin", profit_margin=value) is expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.35, BULLISH), (0.04, NEUTRAL), (-0.10, BEARISH)],
)
def test_revenue_growth_bands(value: float, expected: SignalDirection) -> None:
    assert direction_of("revenue_growth", revenue_growth=value) is expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [(20.0, BULLISH), (90.0, NEUTRAL), (260.0, BEARISH)],
)
def test_debt_to_equity_uses_yahoo_percent_convention(
    value: float, expected: SignalDirection
) -> None:
    assert direction_of("debt_to_equity", debt_to_equity=value) is expected


def test_debt_to_equity_summary_shows_both_units() -> None:
    readings = {reading.name: reading for reading in classify(snapshot(debt_to_equity=145.0))}
    assert "145%" in readings["debt_to_equity"].summary
    assert "1.45x" in readings["debt_to_equity"].summary


def test_missing_metrics_produce_no_readings() -> None:
    assert classify(snapshot()) == ()


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_non_finite_values_are_skipped(value: float) -> None:
    assert classify(snapshot(trailing_pe=value)) == ()


@pytest.mark.parametrize(
    "metric",
    ["price_to_book", "forward_pe", "current_ratio"],
)
def test_non_positive_ratios_are_undefined_not_bearish(metric: str) -> None:
    """A negative book value is a reporting artifact, not a cheap stock."""
    assert classify(snapshot(**{metric: -3.0})) == ()


def test_free_cash_flow_adds_yield_when_market_cap_is_known() -> None:
    readings = {
        reading.name: reading
        for reading in classify(snapshot(free_cash_flow=1.0e10, market_cap=2.0e11))
    }
    reading = readings["free_cash_flow"]
    assert reading.direction is BULLISH
    assert reading.values["fcf_yield_pct"] == pytest.approx(5.0)
    assert "FCF yield" in reading.summary


def test_negative_free_cash_flow_is_bearish() -> None:
    readings = {reading.name: reading for reading in classify(snapshot(free_cash_flow=-5.0e9))}
    assert readings["free_cash_flow"].direction is BEARISH
    assert "burns cash" in readings["free_cash_flow"].summary


def test_forward_pe_compares_against_trailing() -> None:
    readings = {
        reading.name: reading for reading in classify(snapshot(forward_pe=18.0, trailing_pe=25.0))
    }
    summary = readings["forward_pe"].summary
    assert "below the trailing 25.0" in summary
    assert "expects earnings to grow" in summary


def test_readings_carry_categories_and_stable_order() -> None:
    readings = classify(
        snapshot(
            trailing_pe=20.0,
            profit_margin=0.2,
            revenue_growth=0.15,
            debt_to_equity=30.0,
        )
    )
    assert [reading.name for reading in readings] == [
        "trailing_pe",
        "profit_margin",
        "revenue_growth",
        "debt_to_equity",
    ]
    assert readings[0].category is FundamentalCategory.VALUATION
    assert readings[1].category is FundamentalCategory.PROFITABILITY
    assert readings[2].category is FundamentalCategory.GROWTH
    assert readings[3].category is FundamentalCategory.FINANCIAL_HEALTH
