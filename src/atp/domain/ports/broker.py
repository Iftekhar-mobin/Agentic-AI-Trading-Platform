"""Port for order execution venues."""

from __future__ import annotations

from typing import Protocol

from atp.domain.models.orders import Order
from atp.domain.models.trading import TradeProposal


class Broker(Protocol):
    """Executes trades. Paper simulation now; Alpaca/IBKR behind the same port.

    Contract: implementations receive proposals only via ExecuteTrade, which
    calls this exclusively with a risk-approved proposal — the approval is a
    property of the call path, and adapters must not re-derive or bypass it.
    """

    async def submit_market_order(self, proposal: TradeProposal) -> Order: ...
