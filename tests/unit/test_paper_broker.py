"""Tests for the paper broker."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from atp.domain.models.market import Bar, BarInterval, PriceHistory
from atp.domain.models.orders import OrderStatus
from atp.domain.models.trading import OrderSide, TradeProposal
from atp.infrastructure.brokers import PaperBroker

NOW = datetime(2026, 7, 23, 12, 0, tzinfo=UTC)


class StubProvider:
    def __init__(self, price: float | None) -> None:
        self._price = price

    async def get_bars(
        self,
        symbol: str,
        interval: BarInterval,
        *,
        start: datetime,
        end: datetime | None = None,
    ) -> PriceHistory:
        if self._price is None:
            return PriceHistory(symbol=symbol, interval=interval)
        bar = Bar(
            timestamp=NOW - timedelta(days=1),
            open=self._price,
            high=self._price * 1.01,
            low=self._price * 0.99,
            close=self._price,
            volume=1_000.0,
        )
        return PriceHistory(symbol=symbol, interval=interval, bars=(bar,))


def proposal(side: OrderSide = OrderSide.BUY) -> TradeProposal:
    return TradeProposal(
        symbol="AAPL",
        side=side,
        quantity=Decimal("10"),
        entry_price=Decimal("100"),
        stop_loss=Decimal("95"),
    )


def make_broker(price: float | None, slippage_bps: float = 10.0) -> PaperBroker:
    return PaperBroker(
        StubProvider(price),
        slippage_bps=slippage_bps,
        clock=lambda: NOW,
        id_factory=lambda: "test-order-1",
    )


async def test_buy_fills_with_slippage_against_the_trader() -> None:
    order = await make_broker(100.0).submit_market_order(proposal())

    assert order.status is OrderStatus.FILLED
    assert order.fill_price == Decimal("100.1000")  # +10 bps
    assert order.filled_at == NOW
    assert order.id == "test-order-1"
    assert order.stop_loss == Decimal("95")


async def test_sell_fills_below_market() -> None:
    order = await make_broker(100.0).submit_market_order(proposal(OrderSide.SELL))
    assert order.fill_price == Decimal("99.9000")  # -10 bps


async def test_missing_price_rejects_instead_of_raising() -> None:
    order = await make_broker(None).submit_market_order(proposal())
    assert order.status is OrderStatus.REJECTED
    assert order.fill_price is None
    assert order.reason is not None
