"""Yahoo Finance adapter for the FundamentalsProvider port.

Development data source only, with the same caveats as the market-data
adapter: unofficial API, patchy coverage, occasional junk values. Metrics that
are absent, non-numeric, or NaN stay ``None`` — the classification rules then
simply produce no reading for them, which is the honest outcome.

Yahoo's own units are preserved where the domain model documents them
(``debt_to_equity`` as a percentage); everything else is a plain fraction.
"""

from __future__ import annotations

import asyncio
import math
from datetime import UTC, datetime
from typing import Any

import structlog
import yfinance

from atp.domain.models.fundamentals import Fundamentals

log = structlog.get_logger()


def _number(info: dict[str, Any], *keys: str) -> float | None:
    """First key that holds a finite number; vendors rename fields between versions."""
    for key in keys:
        value = info.get(key)
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        number = float(value)
        if math.isfinite(number):
            return number
    return None


def _positive(info: dict[str, Any], *keys: str) -> float | None:
    """Like ``_number`` but rejects non-positive values (the domain model requires > 0)."""
    number = _number(info, *keys)
    return number if number is not None and number > 0 else None


def _text(info: dict[str, Any], key: str) -> str | None:
    value = info.get(key)
    return value.strip() or None if isinstance(value, str) else None


class YFinanceFundamentalsProvider:
    async def get_fundamentals(self, symbol: str) -> Fundamentals:
        symbol = symbol.strip().upper()
        info = await asyncio.to_thread(self._download, symbol)
        return self._to_snapshot(symbol, info)

    @staticmethod
    def _download(symbol: str) -> dict[str, Any]:
        try:
            info = yfinance.Ticker(symbol).get_info()
        except Exception as exc:  # vendor raises bare Exception subclasses
            # An unreachable vendor must not crash a workflow: the caller sees an
            # empty snapshot and reports insufficient data.
            log.warning("fundamentals.fetch_failed", symbol=symbol, error=str(exc))
            return {}
        return info if isinstance(info, dict) else {}

    @staticmethod
    def _to_snapshot(symbol: str, info: dict[str, Any]) -> Fundamentals:
        if not info:
            log.warning("fundamentals.empty_response", symbol=symbol)

        snapshot = Fundamentals(
            symbol=symbol,
            as_of=datetime.now(UTC),
            currency=_text(info, "currency"),
            sector=_text(info, "sector"),
            industry=_text(info, "industry"),
            market_cap=_positive(info, "marketCap"),
            trailing_pe=_number(info, "trailingPE"),
            forward_pe=_number(info, "forwardPE"),
            peg_ratio=_number(info, "trailingPegRatio", "pegRatio"),
            price_to_book=_number(info, "priceToBook"),
            profit_margin=_number(info, "profitMargins"),
            operating_margin=_number(info, "operatingMargins"),
            return_on_equity=_number(info, "returnOnEquity"),
            revenue_growth=_number(info, "revenueGrowth"),
            earnings_growth=_number(info, "earningsGrowth", "earningsQuarterlyGrowth"),
            debt_to_equity=_number(info, "debtToEquity"),
            current_ratio=_number(info, "currentRatio"),
            free_cash_flow=_number(info, "freeCashflow"),
            dividend_yield=_number(info, "dividendYield"),
            beta=_number(info, "beta"),
        )
        populated = sum(
            1
            for field in ("market_cap", "trailing_pe", "profit_margin", "revenue_growth")
            if getattr(snapshot, field) is not None
        )
        log.debug("fundamentals.fetched", symbol=symbol, core_metrics=populated)
        return snapshot
