"""Portfolio and execution-audit endpoints (read scope)."""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel

from atp.domain.models.orders import OrderRecord
from atp.domain.models.trading import Portfolio
from atp.interfaces.api.security import RequiresRead

router = APIRouter(tags=["portfolio"], dependencies=[RequiresRead])


class PositionSchema(BaseModel):
    """Position with its derived numbers computed server-side.

    The properties on the domain model are not serialized by Pydantic, and a
    dashboard recomputing P&L from raw fields is a second implementation of the
    same arithmetic waiting to disagree with the first.
    """

    symbol: str
    quantity: Decimal
    avg_entry_price: Decimal
    current_price: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal
    unrealized_pnl_pct: Decimal


class PortfolioResponse(BaseModel):
    cash: Decimal
    equity: Decimal
    total_market_value: Decimal
    exposure_pct: Decimal
    drawdown_pct: Decimal
    realized_pnl_today: Decimal
    positions: list[PositionSchema]

    @classmethod
    def of(cls, portfolio: Portfolio) -> PortfolioResponse:
        return cls(
            cash=portfolio.cash,
            equity=portfolio.equity,
            total_market_value=portfolio.total_market_value,
            exposure_pct=portfolio.exposure_pct,
            drawdown_pct=portfolio.drawdown_pct,
            realized_pnl_today=portfolio.realized_pnl_today,
            positions=[
                PositionSchema(
                    symbol=position.symbol,
                    quantity=position.quantity,
                    avg_entry_price=position.avg_entry_price,
                    current_price=position.current_price,
                    market_value=position.market_value,
                    unrealized_pnl=position.unrealized_pnl,
                    unrealized_pnl_pct=position.unrealized_pnl_pct,
                )
                for position in portfolio.positions
            ],
        )


class OrdersResponse(BaseModel):
    records: list[OrderRecord]


@router.get("/portfolio", response_model=PortfolioResponse)
async def get_portfolio(request: Request) -> PortfolioResponse:
    """Current paper portfolio, marked to live prices."""
    portfolio = await request.app.state.container.get_portfolio.execute()
    return PortfolioResponse.of(portfolio)


@router.get("/orders", response_model=OrdersResponse)
async def list_orders(
    request: Request,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
) -> OrdersResponse:
    """Execution audit trail: each order with the risk decision that authorized it."""
    records = await request.app.state.container.order_repository.list_records(limit=limit)
    return OrdersResponse(records=records)
