"""Deterministic classification of a fundamentals snapshot.

Same division of labor as the indicator snapshot: fixed, documented thresholds
turn raw metrics into directional readings, and the LLM agent interprets those
readings without ever changing them. Thresholds are broad-market conventions,
not sector-tuned — the agent is told this so it can discount them for, say, a
high-growth name trading on a rich multiple.

A metric the vendor did not supply produces no reading at all.
"""

from __future__ import annotations

import math

from atp.domain.models.analysis import SignalDirection
from atp.domain.models.fundamentals import (
    FundamentalCategory,
    FundamentalReading,
    Fundamentals,
)

BULLISH = SignalDirection.BULLISH
BEARISH = SignalDirection.BEARISH
NEUTRAL = SignalDirection.NEUTRAL

VALUATION = FundamentalCategory.VALUATION
PROFITABILITY = FundamentalCategory.PROFITABILITY
GROWTH = FundamentalCategory.GROWTH
HEALTH = FundamentalCategory.FINANCIAL_HEALTH


def _finite(value: float | None, *, positive: bool = False) -> float | None:
    """Return ``value`` when it is a usable number, else ``None``.

    ``positive=True`` also rejects zero and negatives, for metrics where a
    non-positive reading means "undefined" rather than "bad" (a P/B of -2 is a
    reporting artifact, not a cheap stock).
    """
    if value is None or not math.isfinite(value):
        return None
    if positive and value <= 0:
        return None
    return value


def _banded(
    value: float,
    *,
    bullish_below: float | None = None,
    bearish_above: float | None = None,
    bullish_above: float | None = None,
    bearish_below: float | None = None,
) -> SignalDirection:
    """Classify a value against at most one 'good' and one 'bad' threshold."""
    if bullish_below is not None and value < bullish_below:
        return BULLISH
    if bullish_above is not None and value > bullish_above:
        return BULLISH
    if bearish_above is not None and value > bearish_above:
        return BEARISH
    if bearish_below is not None and value < bearish_below:
        return BEARISH
    return NEUTRAL


def classify(fundamentals: Fundamentals) -> tuple[FundamentalReading, ...]:
    """Return one reading per metric the vendor supplied, in a stable order."""
    readings = [
        _trailing_pe(fundamentals),
        _forward_pe(fundamentals),
        _peg_ratio(fundamentals),
        _price_to_book(fundamentals),
        _profit_margin(fundamentals),
        _operating_margin(fundamentals),
        _return_on_equity(fundamentals),
        _revenue_growth(fundamentals),
        _earnings_growth(fundamentals),
        _debt_to_equity(fundamentals),
        _current_ratio(fundamentals),
        _free_cash_flow(fundamentals),
    ]
    return tuple(reading for reading in readings if reading is not None)


def _trailing_pe(f: Fundamentals) -> FundamentalReading | None:
    if (value := _finite(f.trailing_pe)) is None:
        return None
    if value <= 0:
        # Negative earnings: a P/E is undefined rather than "cheap".
        return FundamentalReading(
            name="trailing_pe",
            category=VALUATION,
            values={"trailing_pe": round(value, 2)},
            direction=BEARISH,
            summary=(
                f"Trailing P/E is {value:.1f} - the company is not profitable on a trailing basis"
            ),
        )
    direction = _banded(value, bullish_below=15.0, bearish_above=30.0)
    return FundamentalReading(
        name="trailing_pe",
        category=VALUATION,
        values={"trailing_pe": round(value, 2)},
        direction=direction,
        summary=f"Trailing P/E is {value:.1f} - {_valuation_note(direction)}",
    )


def _forward_pe(f: Fundamentals) -> FundamentalReading | None:
    if (value := _finite(f.forward_pe, positive=True)) is None:
        return None
    direction = _banded(value, bullish_below=15.0, bearish_above=30.0)
    values = {"forward_pe": round(value, 2)}
    summary = f"Forward P/E is {value:.1f} - {_valuation_note(direction)}"
    trailing = _finite(f.trailing_pe, positive=True)
    if trailing is not None:
        values["trailing_pe"] = round(trailing, 2)
        cheaper = value < trailing
        summary += (
            f"; {'below' if cheaper else 'above'} the trailing {trailing:.1f}, so the "
            f"market expects earnings to {'grow' if cheaper else 'fall'}"
        )
    return FundamentalReading(
        name="forward_pe",
        category=VALUATION,
        values=values,
        direction=direction,
        summary=summary,
    )


def _peg_ratio(f: Fundamentals) -> FundamentalReading | None:
    if (value := _finite(f.peg_ratio)) is None:
        return None
    direction = _banded(value, bullish_below=1.0, bearish_above=2.0) if value > 0 else NEUTRAL
    note = {
        BULLISH: "growth is cheap relative to the multiple",
        BEARISH: "the multiple runs ahead of growth",
        NEUTRAL: "growth is fairly priced",
    }[direction]
    return FundamentalReading(
        name="peg_ratio",
        category=VALUATION,
        values={"peg_ratio": round(value, 2)},
        direction=direction,
        summary=f"PEG is {value:.2f} - {note}",
    )


def _price_to_book(f: Fundamentals) -> FundamentalReading | None:
    if (value := _finite(f.price_to_book, positive=True)) is None:
        return None
    direction = _banded(value, bullish_below=1.5, bearish_above=5.0)
    return FundamentalReading(
        name="price_to_book",
        category=VALUATION,
        values={"price_to_book": round(value, 2)},
        direction=direction,
        summary=f"Price/Book is {value:.2f} - {_valuation_note(direction)}",
    )


def _profit_margin(f: Fundamentals) -> FundamentalReading | None:
    if (value := _finite(f.profit_margin)) is None:
        return None
    direction = _banded(value, bullish_above=0.15, bearish_below=0.05)
    return FundamentalReading(
        name="profit_margin",
        category=PROFITABILITY,
        values={"profit_margin_pct": round(value * 100, 2)},
        direction=direction,
        summary=f"Net margin is {value * 100:.1f}% - {_margin_note(direction)}",
    )


def _operating_margin(f: Fundamentals) -> FundamentalReading | None:
    if (value := _finite(f.operating_margin)) is None:
        return None
    direction = _banded(value, bullish_above=0.20, bearish_below=0.05)
    return FundamentalReading(
        name="operating_margin",
        category=PROFITABILITY,
        values={"operating_margin_pct": round(value * 100, 2)},
        direction=direction,
        summary=f"Operating margin is {value * 100:.1f}% - {_margin_note(direction)}",
    )


def _return_on_equity(f: Fundamentals) -> FundamentalReading | None:
    if (value := _finite(f.return_on_equity)) is None:
        return None
    direction = _banded(value, bullish_above=0.15, bearish_below=0.05)
    note = {
        BULLISH: "capital is being compounded efficiently",
        BEARISH: "weak returns on shareholder capital",
        NEUTRAL: "average returns on shareholder capital",
    }[direction]
    return FundamentalReading(
        name="return_on_equity",
        category=PROFITABILITY,
        values={"return_on_equity_pct": round(value * 100, 2)},
        direction=direction,
        summary=f"ROE is {value * 100:.1f}% - {note}",
    )


def _revenue_growth(f: Fundamentals) -> FundamentalReading | None:
    if (value := _finite(f.revenue_growth)) is None:
        return None
    direction = _banded(value, bullish_above=0.10, bearish_below=0.0)
    return FundamentalReading(
        name="revenue_growth",
        category=GROWTH,
        values={"revenue_growth_pct": round(value * 100, 2)},
        direction=direction,
        summary=(
            f"Revenue growth is {value * 100:.1f}% year over year - {_growth_note(direction)}"
        ),
    )


def _earnings_growth(f: Fundamentals) -> FundamentalReading | None:
    if (value := _finite(f.earnings_growth)) is None:
        return None
    direction = _banded(value, bullish_above=0.10, bearish_below=0.0)
    return FundamentalReading(
        name="earnings_growth",
        category=GROWTH,
        values={"earnings_growth_pct": round(value * 100, 2)},
        direction=direction,
        summary=(
            f"Earnings growth is {value * 100:.1f}% year over year - {_growth_note(direction)}"
        ),
    )


def _debt_to_equity(f: Fundamentals) -> FundamentalReading | None:
    if (value := _finite(f.debt_to_equity)) is None or value < 0:
        return None
    # Yahoo reports debt/equity as a percentage (50 == 0.5x).
    direction = _banded(value, bullish_below=50.0, bearish_above=150.0)
    note = {
        BULLISH: "conservatively financed",
        BEARISH: "carrying heavy leverage",
        NEUTRAL: "moderately levered",
    }[direction]
    return FundamentalReading(
        name="debt_to_equity",
        category=HEALTH,
        values={"debt_to_equity_pct": round(value, 2)},
        direction=direction,
        summary=f"Debt/Equity is {value:.0f}% ({value / 100:.2f}x) - {note}",
    )


def _current_ratio(f: Fundamentals) -> FundamentalReading | None:
    if (value := _finite(f.current_ratio, positive=True)) is None:
        return None
    direction = _banded(value, bullish_above=2.0, bearish_below=1.0)
    note = {
        BULLISH: "comfortable short-term liquidity",
        BEARISH: "current liabilities exceed current assets",
        NEUTRAL: "adequate short-term liquidity",
    }[direction]
    return FundamentalReading(
        name="current_ratio",
        category=HEALTH,
        values={"current_ratio": round(value, 2)},
        direction=direction,
        summary=f"Current ratio is {value:.2f} - {note}",
    )


def _free_cash_flow(f: Fundamentals) -> FundamentalReading | None:
    if (value := _finite(f.free_cash_flow)) is None:
        return None
    direction = BULLISH if value > 0 else BEARISH
    values = {"free_cash_flow": round(value, 2)}
    summary = f"Free cash flow is {value / 1e9:.2f}B - " + (
        "the business self-funds" if direction is BULLISH else "the business burns cash"
    )
    market_cap = _finite(f.market_cap, positive=True)
    if market_cap is not None:
        yield_pct = value / market_cap * 100
        values["fcf_yield_pct"] = round(yield_pct, 2)
        summary += f"; FCF yield {yield_pct:.1f}% of market cap"
    return FundamentalReading(
        name="free_cash_flow",
        category=HEALTH,
        values=values,
        direction=direction,
        summary=summary,
    )


def _valuation_note(direction: SignalDirection) -> str:
    return {
        BULLISH: "cheap against broad-market norms",
        BEARISH: "rich against broad-market norms",
        NEUTRAL: "in line with broad-market norms",
    }[direction]


def _margin_note(direction: SignalDirection) -> str:
    return {
        BULLISH: "strong pricing power",
        BEARISH: "thin profitability",
        NEUTRAL: "ordinary profitability",
    }[direction]


def _growth_note(direction: SignalDirection) -> str:
    return {
        BULLISH: "expanding",
        BEARISH: "contracting",
        NEUTRAL: "roughly flat",
    }[direction]
