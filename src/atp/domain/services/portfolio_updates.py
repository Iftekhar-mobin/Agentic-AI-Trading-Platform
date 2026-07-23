"""Applies order fills to the portfolio. Pure and Decimal-exact.

The risk gate pre-validates affordability and holdings, but this service
guards the same invariants independently — portfolio corruption (negative
cash, phantom shares) must be impossible even if a caller misbehaves.
"""

from __future__ import annotations

from decimal import Decimal

from atp.domain.errors import ExecutionError
from atp.domain.models.orders import Order, OrderStatus
from atp.domain.models.trading import OrderSide, Portfolio, Position


def apply_fill(portfolio: Portfolio, order: Order) -> Portfolio:
    if order.status is not OrderStatus.FILLED or order.fill_price is None:
        msg = f"cannot apply order {order.id} with status '{order.status.value}'"
        raise ExecutionError(msg)
    if order.side is OrderSide.BUY:
        return _apply_buy(portfolio, order, order.fill_price)
    return _apply_sell(portfolio, order, order.fill_price)


def _apply_buy(portfolio: Portfolio, order: Order, fill_price: Decimal) -> Portfolio:
    cost = order.quantity * fill_price
    if cost > portfolio.cash:
        msg = f"fill cost {cost:.2f} exceeds cash {portfolio.cash:.2f}"
        raise ExecutionError(msg)

    existing = portfolio.position_for(order.symbol)
    if existing is None:
        new_position = Position(
            symbol=order.symbol,
            quantity=order.quantity,
            avg_entry_price=fill_price,
            current_price=fill_price,
        )
        positions = (*portfolio.positions, new_position)
    else:
        total_quantity = existing.quantity + order.quantity
        avg_price = (
            existing.quantity * existing.avg_entry_price + order.quantity * fill_price
        ) / total_quantity
        merged = existing.model_copy(
            update={
                "quantity": total_quantity,
                "avg_entry_price": avg_price,
                "current_price": fill_price,
            }
        )
        positions = tuple(
            merged if position.symbol == order.symbol else position
            for position in portfolio.positions
        )
    return portfolio.model_copy(update={"cash": portfolio.cash - cost, "positions": positions})


def _apply_sell(portfolio: Portfolio, order: Order, fill_price: Decimal) -> Portfolio:
    position = portfolio.position_for(order.symbol)
    if position is None:
        msg = f"cannot sell {order.symbol}: no position held"
        raise ExecutionError(msg)
    if order.quantity > position.quantity:
        msg = f"cannot sell {order.quantity} {order.symbol}: only {position.quantity} held"
        raise ExecutionError(msg)

    proceeds = order.quantity * fill_price
    realized = (fill_price - position.avg_entry_price) * order.quantity

    remaining = position.quantity - order.quantity
    if remaining == 0:
        positions = tuple(p for p in portfolio.positions if p.symbol != order.symbol)
    else:
        reduced = position.model_copy(update={"quantity": remaining, "current_price": fill_price})
        positions = tuple(reduced if p.symbol == order.symbol else p for p in portfolio.positions)
    return portfolio.model_copy(
        update={
            "cash": portfolio.cash + proceeds,
            "positions": positions,
            "realized_pnl_today": portfolio.realized_pnl_today + realized,
        }
    )
