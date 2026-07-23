"""Use case: size a prospective trade and run it through the risk gate."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import structlog

from atp.application.use_cases.get_portfolio import GetPortfolio
from atp.domain.errors import InsufficientHistoryError
from atp.domain.models.market import BarInterval
from atp.domain.models.trading import OrderSide, RiskDecision, RiskLimits, TradeProposal
from atp.domain.ports.market_data import MarketDataProvider
from atp.domain.services import (
    evaluate_trade,
    max_affordable,
    size_by_fraction,
    size_by_risk,
)

log = structlog.get_logger()

_PRICE_LOOKBACK = timedelta(days=10)


class CheckTradeRisk:
    def __init__(
        self,
        portfolio: GetPortfolio,
        market_data: MarketDataProvider,
        limits: RiskLimits,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._portfolio = portfolio
        self._market_data = market_data
        self._limits = limits
        self._clock = clock

    async def execute(
        self,
        symbol: str,
        *,
        side: OrderSide = OrderSide.BUY,
        quantity: Decimal | None = None,
        stop_loss: Decimal | None = None,
        take_profit: Decimal | None = None,
        strategy_name: str | None = None,
    ) -> RiskDecision:
        symbol = symbol.strip().upper()
        portfolio = await self._portfolio.execute()
        entry_price = await self._latest_price(symbol)

        if quantity is None:
            if side is OrderSide.SELL:
                position = portfolio.position_for(symbol)
                quantity = position.quantity if position else Decimal("1")
            elif stop_loss is not None:
                # Auto-size to the per-trade risk budget, capped by every limit
                # the sizer can respect up front (cash, position concentration)
                # so the suggested size is actually approvable.
                quantity = min(
                    size_by_risk(
                        portfolio.equity,
                        self._limits.max_risk_per_trade_pct,
                        entry_price,
                        stop_loss,
                    ),
                    size_by_fraction(portfolio.equity, self._limits.max_position_pct, entry_price),
                    max_affordable(portfolio.cash, entry_price),
                )
                quantity = max(quantity, Decimal("1"))  # let the gate explain, not crash
            else:
                quantity = Decimal("1")

        proposal = TradeProposal(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy_name=strategy_name,
        )
        decision = evaluate_trade(proposal, portfolio, self._limits)
        log.info(
            "risk.decision",
            symbol=symbol,
            side=side.value,
            verdict=decision.verdict.value,
            violations=[violation.rule for violation in decision.violations],
        )
        return decision

    async def _latest_price(self, symbol: str) -> Decimal:
        history = await self._market_data.get_bars(
            symbol, BarInterval.DAY_1, start=self._clock() - _PRICE_LOOKBACK
        )
        if history.latest is None:
            msg = f"no recent price available for {symbol}"
            raise InsufficientHistoryError(msg)
        return Decimal(str(history.latest.close))
