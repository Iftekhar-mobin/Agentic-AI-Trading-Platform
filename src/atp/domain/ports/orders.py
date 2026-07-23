"""Port for the execution audit trail."""

from __future__ import annotations

from typing import Protocol

from atp.domain.models.orders import Order, OrderRecord
from atp.domain.models.trading import RiskDecision


class OrderRepository(Protocol):
    """Append-only store of orders with the risk decisions that authorized them."""

    async def append(self, order: Order, decision: RiskDecision) -> None: ...

    async def list_records(self, *, limit: int | None = None) -> list[OrderRecord]: ...
