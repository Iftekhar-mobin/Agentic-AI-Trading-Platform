"""Order models and the execution result.

Orders exist only downstream of an approved RiskDecision — the application
provides no path that creates one otherwise. The lifecycle is deliberately
small for market orders: submitted -> filled | rejected; cancellation and
partial fills arrive with real broker adapters.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from atp.domain.models.trading import OrderSide, RiskDecision


class OrderStatus(StrEnum):
    SUBMITTED = "submitted"
    FILLED = "filled"
    REJECTED = "rejected"


class Order(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1)
    symbol: str = Field(min_length=1, max_length=12)
    side: OrderSide
    quantity: Decimal = Field(gt=0)
    order_type: Literal["market"] = "market"
    stop_loss: Decimal | None = Field(default=None, gt=0)
    take_profit: Decimal | None = Field(default=None, gt=0)
    strategy_name: str | None = None
    status: OrderStatus
    submitted_at: datetime
    fill_price: Decimal | None = Field(default=None, gt=0)
    filled_at: datetime | None = None
    reason: str | None = Field(default=None, description="Broker rejection reason")

    @model_validator(mode="after")
    def _check_fill_consistency(self) -> Order:
        if self.status is OrderStatus.FILLED and (
            self.fill_price is None or self.filled_at is None
        ):
            msg = "a filled order must carry fill_price and filled_at"
            raise ValueError(msg)
        if self.status is not OrderStatus.FILLED and self.fill_price is not None:
            msg = f"order with status '{self.status.value}' cannot carry a fill price"
            raise ValueError(msg)
        return self

    @property
    def notional(self) -> Decimal | None:
        return self.quantity * self.fill_price if self.fill_price is not None else None


class OrderRecord(BaseModel):
    """One line of the execution audit trail: the order and the risk decision
    that authorized it, stored together so neither can be interpreted alone."""

    model_config = ConfigDict(frozen=True)

    order: Order
    decision: RiskDecision
    recorded_at: datetime


class ExecutionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    decision: RiskDecision
    order: Order | None  # None when the risk gate rejected the proposal

    @property
    def executed(self) -> bool:
        return self.order is not None and self.order.status is OrderStatus.FILLED
