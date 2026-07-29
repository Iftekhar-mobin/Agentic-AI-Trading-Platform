"""Risk-check and trade-execution endpoints.

Two endpoints with deliberately different scopes: sizing a trade and seeing
what the risk gate would say is a *read* operation and safe to expose to a
dashboard, while actually submitting it requires the ``trade`` scope.

Every execution — fills and gate rejections alike — is journalled to episodic
memory here, exactly as the CLI does, so the trade history a later analysis
recalls does not depend on which interface placed the order.
"""

from __future__ import annotations

from decimal import Decimal

from fastapi import APIRouter, Request, Response, status
from pydantic import BaseModel, Field

from atp.domain.models.orders import ExecutionResult, Order
from atp.domain.models.trading import OrderSide, RiskDecision
from atp.interfaces.api.security import RequiresRead, RequiresTrade

router = APIRouter(tags=["trading"])


class TradeRequest(BaseModel):
    symbol: str = Field(min_length=1, max_length=12, examples=["AAPL"])
    side: OrderSide = OrderSide.BUY
    quantity: Decimal | None = Field(
        default=None,
        gt=0,
        description="Omit to let the risk engine size the position from your limits",
    )
    stop_loss: Decimal | None = Field(default=None, gt=0)
    take_profit: Decimal | None = Field(default=None, gt=0)
    strategy_name: str | None = None


class RiskCheckResponse(BaseModel):
    decision: RiskDecision


class TradeResponse(BaseModel):
    decision: RiskDecision
    order: Order | None = None
    executed: bool


@router.post("/risk-check", response_model=RiskCheckResponse, dependencies=[RequiresRead])
async def risk_check(request: TradeRequest, http_request: Request) -> RiskCheckResponse:
    """Size a proposal and run it through the risk gate without submitting anything."""
    decision = await http_request.app.state.container.check_trade_risk.execute(
        request.symbol,
        side=request.side,
        quantity=request.quantity,
        stop_loss=request.stop_loss,
        take_profit=request.take_profit,
        strategy_name=request.strategy_name,
    )
    return RiskCheckResponse(decision=decision)


@router.post("/trade", response_model=TradeResponse, dependencies=[RequiresTrade])
async def place_trade(
    request: TradeRequest, http_request: Request, response: Response
) -> TradeResponse:
    """Execute through the risk gate. A rejected proposal returns 200 with the reasons.

    Not an error status: the gate refusing a trade is the system working, and
    the caller needs the decision body either way. ``executed`` and the order's
    status carry the outcome.
    """
    container = http_request.app.state.container
    result: ExecutionResult = await container.execute_trade.execute(
        request.symbol,
        side=request.side,
        quantity=request.quantity,
        stop_loss=request.stop_loss,
        take_profit=request.take_profit,
        strategy_name=request.strategy_name,
    )
    await container.journal_trade.execute(result)

    if result.order is not None and result.executed:
        response.status_code = status.HTTP_201_CREATED
    return TradeResponse(decision=result.decision, order=result.order, executed=result.executed)
