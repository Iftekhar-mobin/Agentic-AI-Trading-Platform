"""Fundamental analysis domain models.

Mirrors the technical stack deliberately: a vendor snapshot is turned into
deterministic, rule-classified ``FundamentalReading``s, and only then does an
LLM interpret them. Every field on the snapshot is optional because vendor
fundamentals are patchy — a missing metric produces no reading rather than a
guessed one.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from atp.domain.models.analysis import SignalDirection
from atp.domain.models.explainability import Explanation

MIN_FUNDAMENTAL_READINGS = 3
"""Below this many classified metrics a fundamental view is not worth forming."""


class FundamentalCategory(StrEnum):
    VALUATION = "valuation"
    PROFITABILITY = "profitability"
    GROWTH = "growth"
    FINANCIAL_HEALTH = "financial_health"


class Fundamentals(BaseModel):
    """Raw vendor snapshot, normalized to canonical units.

    Ratios that vendors report as fractions (margins, growth, yields) are kept
    as fractions here; ``debt_to_equity`` keeps Yahoo's percentage convention
    and is documented as such at every use site.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str = Field(min_length=1)
    as_of: datetime
    currency: str | None = None
    sector: str | None = None
    industry: str | None = None

    market_cap: float | None = Field(default=None, gt=0)
    trailing_pe: float | None = None
    forward_pe: float | None = None
    peg_ratio: float | None = None
    price_to_book: float | None = None

    profit_margin: float | None = None
    operating_margin: float | None = None
    return_on_equity: float | None = None

    revenue_growth: float | None = None
    earnings_growth: float | None = None

    debt_to_equity: float | None = Field(default=None, description="Percent, Yahoo convention")
    current_ratio: float | None = None
    free_cash_flow: float | None = None

    dividend_yield: float | None = None
    beta: float | None = None

    @field_validator("symbol")
    @classmethod
    def _normalize_symbol(cls, value: str) -> str:
        return value.strip().upper()

    @field_validator("as_of")
    @classmethod
    def _normalize_to_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            msg = "fundamentals timestamps must be timezone-aware"
            raise ValueError(msg)
        return value.astimezone(UTC)


class FundamentalReading(BaseModel):
    """One metric's value plus its deterministic, rule-based classification."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1, description="Stable identifier, e.g. 'trailing_pe'")
    category: FundamentalCategory
    values: dict[str, float]
    direction: SignalDirection
    summary: str = Field(min_length=1, description="Human-readable one-liner with the numbers")


class FundamentalAssessment(Explanation):
    """The LLM's interpretation of the deterministic readings."""

    direction: SignalDirection


class FundamentalReport(BaseModel):
    """Full output of the Fundamental Analysis agent."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    as_of: datetime
    fundamentals: Fundamentals
    readings: tuple[FundamentalReading, ...]
    signal_counts: dict[SignalDirection, int]
    assessment: FundamentalAssessment
