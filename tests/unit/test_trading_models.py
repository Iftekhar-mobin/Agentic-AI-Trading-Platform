"""Tests for trading domain models (Decimal arithmetic)."""

from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from atp.domain.models.trading import Portfolio, Position, TradeProposal


def make_position(
    symbol: str = "AAPL",
    quantity: str = "10",
    entry: str = "100",
    price: str = "110",
) -> Position:
    return Position(
        symbol=symbol,
        quantity=Decimal(quantity),
        avg_entry_price=Decimal(entry),
        current_price=Decimal(price),
    )


class TestPosition:
    def test_valuation_math(self) -> None:
        position = make_position()
        assert position.market_value == Decimal("1100")
        assert position.unrealized_pnl == Decimal("100")
        assert position.unrealized_pnl_pct == Decimal("10")


class TestPortfolio:
    def test_equity_and_exposure(self) -> None:
        portfolio = Portfolio(
            cash=Decimal("5000"),
            positions=(make_position(), make_position("MSFT", "5", "200", "220")),
        )
        assert portfolio.total_market_value == Decimal("2200")
        assert portfolio.equity == Decimal("7200")
        assert float(portfolio.exposure_pct) == pytest.approx(2200 / 7200 * 100)

    def test_duplicate_symbols_rejected(self) -> None:
        with pytest.raises(ValidationError, match="duplicate"):
            Portfolio(cash=Decimal("0"), positions=(make_position(), make_position()))

    def test_with_prices_updates_only_named_symbols(self) -> None:
        portfolio = Portfolio(
            cash=Decimal("0"),
            positions=(make_position(), make_position("MSFT", "5", "200", "220")),
        )
        updated = portfolio.with_prices({"AAPL": Decimal("120")})
        aapl = updated.position_for("AAPL")
        msft = updated.position_for("MSFT")
        assert aapl is not None
        assert aapl.current_price == Decimal("120")
        assert msft is not None
        assert msft.current_price == Decimal("220")

    def test_peak_and_drawdown(self) -> None:
        portfolio = Portfolio(cash=Decimal("10000")).with_peak_updated()
        assert portfolio.equity_peak == Decimal("10000")
        assert portfolio.drawdown_pct == Decimal("0")

        # Simulate losing 1000 of equity with a fixed prior peak.
        drawn_down = Portfolio(cash=Decimal("9000"), equity_peak=Decimal("10000"))
        assert drawn_down.drawdown_pct == Decimal("10")
        # Peak never decreases.
        assert drawn_down.with_peak_updated().equity_peak == Decimal("10000")

    def test_position_lookup_normalizes_symbol(self) -> None:
        portfolio = Portfolio(cash=Decimal("0"), positions=(make_position(),))
        assert portfolio.position_for(" aapl ") is not None


class TestTradeProposal:
    def test_notional(self) -> None:
        proposal = TradeProposal(
            symbol="aapl",
            side="buy",
            quantity=Decimal("10"),
            entry_price=Decimal("325.50"),
        )
        assert proposal.symbol == "AAPL"
        assert proposal.notional == Decimal("3255.00")

    def test_json_round_trip_preserves_decimals(self) -> None:
        proposal = TradeProposal(
            symbol="AAPL",
            side="buy",
            quantity=Decimal("10"),
            entry_price=Decimal("325.53"),
            stop_loss=Decimal("310.10"),
        )
        restored = TradeProposal.model_validate_json(proposal.model_dump_json())
        assert restored == proposal
        assert restored.entry_price == Decimal("325.53")
