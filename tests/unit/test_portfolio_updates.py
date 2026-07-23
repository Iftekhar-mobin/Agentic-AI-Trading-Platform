"""Golden-value tests for applying fills to the portfolio."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from atp.domain.errors import ExecutionError
from atp.domain.models.orders import Order, OrderStatus
from atp.domain.models.trading import OrderSide, Portfolio, Position
from atp.domain.services import apply_fill

NOW = datetime(2026, 7, 23, tzinfo=UTC)


def order(
    side: OrderSide,
    quantity: str,
    fill: str,
    symbol: str = "AAPL",
    status: OrderStatus = OrderStatus.FILLED,
) -> Order:
    return Order(
        id="ord-1",
        symbol=symbol,
        side=side,
        quantity=Decimal(quantity),
        status=status,
        submitted_at=NOW,
        fill_price=Decimal(fill) if status is OrderStatus.FILLED else None,
        filled_at=NOW if status is OrderStatus.FILLED else None,
    )


def holding(quantity: str = "10", entry: str = "100") -> Position:
    return Position(
        symbol="AAPL",
        quantity=Decimal(quantity),
        avg_entry_price=Decimal(entry),
        current_price=Decimal(entry),
    )


class TestBuyFills:
    def test_opens_new_position(self) -> None:
        portfolio = Portfolio(cash=Decimal("10000"))
        updated = apply_fill(portfolio, order(OrderSide.BUY, "10", "100"))

        assert updated.cash == Decimal("9000")
        position = updated.position_for("AAPL")
        assert position is not None
        assert position.quantity == Decimal("10")
        assert position.avg_entry_price == Decimal("100")

    def test_averages_into_existing_position(self) -> None:
        portfolio = Portfolio(cash=Decimal("10000"), positions=(holding("10", "100"),))
        updated = apply_fill(portfolio, order(OrderSide.BUY, "10", "120"))

        position = updated.position_for("AAPL")
        assert position is not None
        assert position.quantity == Decimal("20")
        assert position.avg_entry_price == Decimal("110")  # (10*100 + 10*120)/20
        assert updated.cash == Decimal("8800")

    def test_insufficient_cash_is_execution_error(self) -> None:
        portfolio = Portfolio(cash=Decimal("100"))
        with pytest.raises(ExecutionError, match="exceeds cash"):
            apply_fill(portfolio, order(OrderSide.BUY, "10", "100"))


class TestSellFills:
    def test_partial_sell_realizes_pnl(self) -> None:
        portfolio = Portfolio(cash=Decimal("0"), positions=(holding("10", "100"),))
        updated = apply_fill(portfolio, order(OrderSide.SELL, "4", "110"))

        assert updated.cash == Decimal("440")
        assert updated.realized_pnl_today == Decimal("40")  # (110-100)*4
        position = updated.position_for("AAPL")
        assert position is not None
        assert position.quantity == Decimal("6")
        assert position.avg_entry_price == Decimal("100")  # basis unchanged

    def test_full_sell_removes_position(self) -> None:
        portfolio = Portfolio(cash=Decimal("0"), positions=(holding("10", "100"),))
        updated = apply_fill(portfolio, order(OrderSide.SELL, "10", "90"))

        assert updated.position_for("AAPL") is None
        assert updated.cash == Decimal("900")
        assert updated.realized_pnl_today == Decimal("-100")

    def test_oversell_is_execution_error(self) -> None:
        portfolio = Portfolio(cash=Decimal("0"), positions=(holding("10", "100"),))
        with pytest.raises(ExecutionError, match="only 10"):
            apply_fill(portfolio, order(OrderSide.SELL, "11", "110"))

    def test_sell_without_position_is_execution_error(self) -> None:
        with pytest.raises(ExecutionError, match="no position"):
            apply_fill(Portfolio(cash=Decimal("0")), order(OrderSide.SELL, "1", "110"))


class TestGuards:
    def test_unfilled_order_rejected(self) -> None:
        unfilled = order(OrderSide.BUY, "10", "100", status=OrderStatus.REJECTED)
        with pytest.raises(ExecutionError, match="rejected"):
            apply_fill(Portfolio(cash=Decimal("10000")), unfilled)
