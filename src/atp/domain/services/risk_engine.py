"""The risk gate: deterministic evaluation of a trade proposal.

This is the platform's veto. It is pure arithmetic over the proposal, the
portfolio, and the configured limits — no LLM is consulted, ever. It reports
*every* violated rule (not just the first) plus all evaluated metrics, so a
rejection is fully explainable and auditable.

Sells are risk-reducing and are exempt from entry limits (including the daily
loss and drawdown halts — a halted account must still be able to exit), but
must reference real holdings: short selling is not supported.
"""

from __future__ import annotations

from decimal import Decimal

from atp.domain.models.trading import (
    OrderSide,
    Portfolio,
    RiskDecision,
    RiskLimits,
    RiskVerdict,
    RiskViolation,
    TradeProposal,
)


def evaluate_trade(
    proposal: TradeProposal,
    portfolio: Portfolio,
    limits: RiskLimits,
) -> RiskDecision:
    violations: list[RiskViolation] = []
    metrics: dict[str, float] = {}

    equity = portfolio.equity
    metrics["equity"] = float(equity)
    metrics["cash"] = float(portfolio.cash)

    if equity <= 0:
        violations.append(RiskViolation(rule="solvency", detail="portfolio equity is not positive"))
        return _decision(proposal, limits, violations, metrics)

    if proposal.side is OrderSide.SELL:
        _check_sell(proposal, portfolio, violations)
        return _decision(proposal, limits, violations, metrics)

    _check_halts(portfolio, limits, violations, metrics)
    _check_cash(proposal, portfolio, violations, metrics)
    _check_stop_and_risk(proposal, equity, limits, violations, metrics)
    _check_concentration(proposal, portfolio, equity, limits, violations, metrics)
    _check_position_count(proposal, portfolio, limits, violations, metrics)
    return _decision(proposal, limits, violations, metrics)


def _decision(
    proposal: TradeProposal,
    limits: RiskLimits,
    violations: list[RiskViolation],
    metrics: dict[str, float],
) -> RiskDecision:
    return RiskDecision(
        verdict=RiskVerdict.REJECTED if violations else RiskVerdict.APPROVED,
        proposal=proposal,
        limits=limits,
        violations=tuple(violations),
        metrics=metrics,
    )


def _check_sell(
    proposal: TradeProposal, portfolio: Portfolio, violations: list[RiskViolation]
) -> None:
    position = portfolio.position_for(proposal.symbol)
    if position is None:
        violations.append(
            RiskViolation(
                rule="no_position",
                detail=f"no {proposal.symbol} position held; short selling is not supported",
            )
        )
    elif proposal.quantity > position.quantity:
        violations.append(
            RiskViolation(
                rule="oversell",
                detail=(f"selling {proposal.quantity} exceeds held quantity {position.quantity}"),
            )
        )


def _check_halts(
    portfolio: Portfolio,
    limits: RiskLimits,
    violations: list[RiskViolation],
    metrics: dict[str, float],
) -> None:
    daily_loss_pct = (
        float(-portfolio.realized_pnl_today / portfolio.equity * 100)
        if portfolio.realized_pnl_today < 0
        else 0.0
    )
    metrics["daily_loss_pct"] = daily_loss_pct
    if daily_loss_pct >= limits.max_daily_loss_pct:
        violations.append(
            RiskViolation(
                rule="daily_loss_halt",
                detail=(
                    f"today's realized loss {daily_loss_pct:.2f}% >= "
                    f"limit {limits.max_daily_loss_pct}%; new entries halted"
                ),
            )
        )

    drawdown_pct = float(portfolio.drawdown_pct)
    metrics["drawdown_pct"] = drawdown_pct
    if drawdown_pct >= limits.max_drawdown_pct:
        violations.append(
            RiskViolation(
                rule="drawdown_halt",
                detail=(
                    f"drawdown {drawdown_pct:.2f}% >= limit "
                    f"{limits.max_drawdown_pct}%; new entries halted"
                ),
            )
        )


def _check_cash(
    proposal: TradeProposal,
    portfolio: Portfolio,
    violations: list[RiskViolation],
    metrics: dict[str, float],
) -> None:
    cost = proposal.notional
    metrics["cost"] = float(cost)
    if cost > portfolio.cash:
        violations.append(
            RiskViolation(
                rule="insufficient_cash",
                detail=f"cost {cost:.2f} exceeds available cash {portfolio.cash:.2f}",
            )
        )


def _check_stop_and_risk(
    proposal: TradeProposal,
    equity: Decimal,
    limits: RiskLimits,
    violations: list[RiskViolation],
    metrics: dict[str, float],
) -> None:
    if proposal.stop_loss is None:
        if limits.require_stop_loss:
            violations.append(
                RiskViolation(rule="missing_stop_loss", detail="a stop loss is required")
            )
        return
    if proposal.stop_loss >= proposal.entry_price:
        violations.append(
            RiskViolation(
                rule="invalid_stop",
                detail=(
                    f"stop {proposal.stop_loss} must be below entry "
                    f"{proposal.entry_price} for a long"
                ),
            )
        )
        return
    risk_amount = (proposal.entry_price - proposal.stop_loss) * proposal.quantity
    risk_pct = float(risk_amount / equity * 100)
    metrics["risk_amount"] = float(risk_amount)
    metrics["risk_pct"] = risk_pct
    if risk_pct > limits.max_risk_per_trade_pct:
        violations.append(
            RiskViolation(
                rule="risk_per_trade",
                detail=(
                    f"stop-based risk {risk_pct:.2f}% exceeds "
                    f"limit {limits.max_risk_per_trade_pct}%"
                ),
            )
        )


def _check_concentration(
    proposal: TradeProposal,
    portfolio: Portfolio,
    equity: Decimal,
    limits: RiskLimits,
    violations: list[RiskViolation],
    metrics: dict[str, float],
) -> None:
    existing = portfolio.position_for(proposal.symbol)
    existing_value = existing.market_value if existing else Decimal("0")
    position_pct = float((existing_value + proposal.notional) / equity * 100)
    metrics["position_pct_after"] = position_pct
    if position_pct > limits.max_position_pct:
        violations.append(
            RiskViolation(
                rule="position_concentration",
                detail=(
                    f"{proposal.symbol} would be {position_pct:.2f}% of equity, "
                    f"limit {limits.max_position_pct}%"
                ),
            )
        )

    exposure_pct = float((portfolio.total_market_value + proposal.notional) / equity * 100)
    metrics["exposure_pct_after"] = exposure_pct
    if exposure_pct > limits.max_total_exposure_pct:
        violations.append(
            RiskViolation(
                rule="total_exposure",
                detail=(
                    f"total exposure would be {exposure_pct:.2f}%, "
                    f"limit {limits.max_total_exposure_pct}%"
                ),
            )
        )


def _check_position_count(
    proposal: TradeProposal,
    portfolio: Portfolio,
    limits: RiskLimits,
    violations: list[RiskViolation],
    metrics: dict[str, float],
) -> None:
    metrics["open_positions"] = float(len(portfolio.positions))
    is_new_symbol = portfolio.position_for(proposal.symbol) is None
    if is_new_symbol and len(portfolio.positions) >= limits.max_positions:
        violations.append(
            RiskViolation(
                rule="max_positions",
                detail=(
                    f"already holding {len(portfolio.positions)} positions, "
                    f"limit {limits.max_positions}"
                ),
            )
        )
