"""Local paper broker: simulated market-order fills at the latest price.

Fills apply configurable slippage against the trader (buys fill higher, sells
lower) so paper results stay conservative. A missing price yields a REJECTED
order rather than an exception — brokers report outcomes, they don't crash
workflows.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import structlog

from atp.domain.models.market import BarInterval
from atp.domain.models.orders import Order, OrderStatus
from atp.domain.models.trading import OrderSide, TradeProposal
from atp.domain.ports.market_data import MarketDataProvider

log = structlog.get_logger()

_PRICE_LOOKBACK = timedelta(days=10)


class PaperBroker:
    def __init__(
        self,
        market_data: MarketDataProvider,
        *,
        slippage_bps: float = 5.0,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        id_factory: Callable[[], str] = lambda: str(uuid.uuid4()),
    ) -> None:
        self._market_data = market_data
        self._slippage = Decimal(str(slippage_bps)) / 10_000
        self._clock = clock
        self._id_factory = id_factory

    async def submit_market_order(self, proposal: TradeProposal) -> Order:
        now = self._clock()
        base = {
            "id": self._id_factory(),
            "symbol": proposal.symbol,
            "side": proposal.side,
            "quantity": proposal.quantity,
            "stop_loss": proposal.stop_loss,
            "take_profit": proposal.take_profit,
            "strategy_name": proposal.strategy_name,
            "submitted_at": now,
        }

        price = await self._latest_price(proposal.symbol)
        if price is None:
            log.warning("broker.rejected", symbol=proposal.symbol, reason="no price")
            return Order(**base, status=OrderStatus.REJECTED, reason="no recent price available")

        if proposal.side is OrderSide.BUY:
            fill_price = price * (1 + self._slippage)
        else:
            fill_price = price * (1 - self._slippage)

        order = Order(
            **base,
            status=OrderStatus.FILLED,
            fill_price=fill_price.quantize(Decimal("0.0001")),
            filled_at=now,
        )
        log.info(
            "broker.filled",
            order_id=order.id,
            symbol=order.symbol,
            side=order.side.value,
            quantity=float(order.quantity),
            fill_price=float(order.fill_price or 0),
        )
        return order

    async def _latest_price(self, symbol: str) -> Decimal | None:
        history = await self._market_data.get_bars(
            symbol, BarInterval.DAY_1, start=self._clock() - _PRICE_LOOKBACK
        )
        if history.latest is None:
            return None
        return Decimal(str(history.latest.close))
