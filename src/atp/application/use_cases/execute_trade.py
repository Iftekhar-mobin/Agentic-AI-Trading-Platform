"""Use case: execute a trade through the risk gate.

This is the ONLY code path that reaches the Broker port, and the broker call
sits strictly behind the risk approval branch. The gate is therefore
structural: there is no sequence of calls in this codebase that submits an
order the risk engine rejected.

Every outcome — rejection included, as a decision with no order — lands in
the append-only audit trail with the risk decision that produced it.
"""

from __future__ import annotations

from decimal import Decimal

import structlog

from atp.application.use_cases.check_trade_risk import CheckTradeRisk
from atp.domain.errors import ExecutionError
from atp.domain.models.orders import ExecutionResult, OrderStatus
from atp.domain.models.trading import OrderSide
from atp.domain.ports.broker import Broker
from atp.domain.ports.orders import OrderRepository
from atp.domain.ports.portfolio import PortfolioRepository
from atp.domain.services.portfolio_updates import apply_fill

log = structlog.get_logger()


class ExecuteTrade:
    def __init__(
        self,
        check_trade_risk: CheckTradeRisk,
        broker: Broker,
        portfolio_repository: PortfolioRepository,
        order_repository: OrderRepository,
    ) -> None:
        self._check_trade_risk = check_trade_risk
        self._broker = broker
        self._portfolio_repository = portfolio_repository
        self._order_repository = order_repository

    async def execute(
        self,
        symbol: str,
        *,
        side: OrderSide = OrderSide.BUY,
        quantity: Decimal | None = None,
        stop_loss: Decimal | None = None,
        take_profit: Decimal | None = None,
        strategy_name: str | None = None,
    ) -> ExecutionResult:
        decision = await self._check_trade_risk.execute(
            symbol,
            side=side,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy_name=strategy_name,
        )

        if not decision.approved:
            log.warning(
                "execution.blocked_by_risk_gate",
                symbol=decision.proposal.symbol,
                violations=[violation.rule for violation in decision.violations],
            )
            return ExecutionResult(decision=decision, order=None)

        # Reached only with an approved decision — the gate is this branch.
        order = await self._broker.submit_market_order(decision.proposal)

        if order.status is OrderStatus.FILLED:
            portfolio = await self._portfolio_repository.load()
            try:
                portfolio = apply_fill(portfolio, order).with_peak_updated()
            except ExecutionError:
                # Audit the fill before surfacing the inconsistency.
                await self._order_repository.append(order, decision)
                raise
            await self._portfolio_repository.save(portfolio)
            log.info(
                "execution.filled",
                order_id=order.id,
                symbol=order.symbol,
                side=order.side.value,
                fill_price=float(order.fill_price or 0),
                cash_after=float(portfolio.cash),
                equity_after=float(portfolio.equity),
            )
        else:
            log.warning("execution.broker_rejected", order_id=order.id, reason=order.reason)

        await self._order_repository.append(order, decision)
        return ExecutionResult(decision=decision, order=order)
