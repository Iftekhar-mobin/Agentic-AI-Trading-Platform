"""Tests for ExecuteTrade — including the structural risk-gate guarantee."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from atp.application.use_cases import CheckTradeRisk, ExecuteTrade, GetPortfolio
from atp.domain.models.market import Bar, BarInterval, PriceHistory
from atp.domain.models.orders import OrderStatus
from atp.domain.models.trading import OrderSide, RiskLimits, TradeProposal
from atp.infrastructure.brokers import PaperBroker
from atp.infrastructure.persistence.json_orders import JsonOrderRepository
from atp.infrastructure.persistence.json_portfolio import JsonPortfolioRepository

NOW = datetime(2026, 7, 23, 12, 0, tzinfo=UTC)


class StubProvider:
    def __init__(self, price: float) -> None:
        self._price = price

    async def get_bars(
        self,
        symbol: str,
        interval: BarInterval,
        *,
        start: datetime,
        end: datetime | None = None,
    ) -> PriceHistory:
        bar = Bar(
            timestamp=NOW - timedelta(days=1),
            open=self._price,
            high=self._price * 1.01,
            low=self._price * 0.99,
            close=self._price,
            volume=1_000.0,
        )
        return PriceHistory(symbol=symbol, interval=interval, bars=(bar,))


class SpyBroker:
    """Records every submission; the gate test asserts it stays untouched."""

    def __init__(self, inner: PaperBroker) -> None:
        self._inner = inner
        self.submissions: list[TradeProposal] = []

    async def submit_market_order(self, proposal: TradeProposal):  # type: ignore[no-untyped-def]
        self.submissions.append(proposal)
        return await self._inner.submit_market_order(proposal)


def build(
    tmp_path: Path, price: float = 100.0
) -> tuple[ExecuteTrade, SpyBroker, JsonPortfolioRepository, JsonOrderRepository]:
    provider = StubProvider(price)
    portfolio_repository = JsonPortfolioRepository(
        tmp_path / "portfolio.json", starting_cash=Decimal("100000")
    )
    order_repository = JsonOrderRepository(tmp_path / "orders.jsonl", clock=lambda: NOW)
    broker = SpyBroker(PaperBroker(provider, slippage_bps=0.0, clock=lambda: NOW))
    check = CheckTradeRisk(
        GetPortfolio(portfolio_repository, provider, clock=lambda: NOW),
        provider,
        RiskLimits(),
        clock=lambda: NOW,
    )
    use_case = ExecuteTrade(check, broker, portfolio_repository, order_repository)
    return use_case, broker, portfolio_repository, order_repository


async def test_approved_trade_fills_and_updates_portfolio(tmp_path: Path) -> None:
    use_case, broker, portfolio_repository, order_repository = build(tmp_path)

    result = await use_case.execute("AAPL", stop_loss=Decimal("95"))

    assert result.executed
    assert result.order is not None
    assert result.order.status is OrderStatus.FILLED
    assert len(broker.submissions) == 1

    portfolio = await portfolio_repository.load()
    position = portfolio.position_for("AAPL")
    assert position is not None
    assert position.quantity == result.decision.proposal.quantity
    assert portfolio.cash == Decimal("100000") - (result.order.fill_price or 0) * position.quantity

    records = await order_repository.list_records()
    assert len(records) == 1
    assert records[0].order.id == result.order.id
    assert records[0].decision.approved


async def test_rejected_trade_never_reaches_the_broker(tmp_path: Path) -> None:
    """THE structural guarantee: no approval, no broker call, no order."""
    use_case, broker, portfolio_repository, _ = build(tmp_path)

    result = await use_case.execute("AAPL")  # no stop loss -> rejected

    assert not result.executed
    assert result.order is None
    assert broker.submissions == []  # the broker was never touched

    portfolio = await portfolio_repository.load()
    assert portfolio.cash == Decimal("100000")
    assert portfolio.positions == ()


async def test_round_trip_buy_then_sell_realizes_pnl(tmp_path: Path) -> None:
    use_case, _, portfolio_repository, order_repository = build(tmp_path)

    buy = await use_case.execute("AAPL", stop_loss=Decimal("95"))
    assert buy.executed

    sell = await use_case.execute("AAPL", side=OrderSide.SELL)
    assert sell.executed

    portfolio = await portfolio_repository.load()
    assert portfolio.positions == ()  # flat again
    assert portfolio.cash == Decimal("100000")  # zero slippage round trip
    assert len(await order_repository.list_records()) == 2


async def test_sell_without_position_is_gated_not_brokered(tmp_path: Path) -> None:
    use_case, broker, _, _ = build(tmp_path)

    result = await use_case.execute("TSLA", side=OrderSide.SELL)

    assert not result.executed
    assert broker.submissions == []
    assert "no_position" in [v.rule for v in result.decision.violations]
