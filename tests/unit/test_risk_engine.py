"""Tests for the risk gate: every rule, approval and rejection paths."""

from __future__ import annotations

from decimal import Decimal

from atp.domain.models.trading import (
    OrderSide,
    Portfolio,
    Position,
    RiskLimits,
    TradeProposal,
)
from atp.domain.services import evaluate_trade

LIMITS = RiskLimits()  # defaults: 1% risk, 20% position, 100% exposure, 10 pos, 3%/20% halts


def make_portfolio(
    cash: str = "100000",
    positions: tuple[Position, ...] = (),
    equity_peak: str | None = None,
    realized_pnl_today: str = "0",
) -> Portfolio:
    return Portfolio(
        cash=Decimal(cash),
        positions=positions,
        equity_peak=Decimal(equity_peak) if equity_peak else None,
        realized_pnl_today=Decimal(realized_pnl_today),
    )


def buy(
    quantity: str = "100",
    entry: str = "100",
    stop: str | None = "99",
    symbol: str = "AAPL",
) -> TradeProposal:
    return TradeProposal(
        symbol=symbol,
        side=OrderSide.BUY,
        quantity=Decimal(quantity),
        entry_price=Decimal(entry),
        stop_loss=Decimal(stop) if stop else None,
    )


def sell(quantity: str, symbol: str = "AAPL") -> TradeProposal:
    return TradeProposal(
        symbol=symbol,
        side=OrderSide.SELL,
        quantity=Decimal(quantity),
        entry_price=Decimal("100"),
    )


def position(symbol: str = "AAPL", quantity: str = "50", price: str = "100") -> Position:
    return Position(
        symbol=symbol,
        quantity=Decimal(quantity),
        avg_entry_price=Decimal(price),
        current_price=Decimal(price),
    )


class TestApproval:
    def test_sane_trade_is_approved_with_metrics(self) -> None:
        # 100 shares, $1 stop distance = $100 risk = 0.1% of 100k. Well inside.
        decision = evaluate_trade(buy(), make_portfolio(), LIMITS)

        assert decision.approved
        assert decision.violations == ()
        assert decision.metrics["risk_pct"] == 0.1
        assert decision.metrics["cost"] == 10_000.0
        assert decision.metrics["position_pct_after"] == 10.0
        assert decision.metrics["exposure_pct_after"] == 10.0

    def test_decision_embeds_proposal_and_limits(self) -> None:
        decision = evaluate_trade(buy(), make_portfolio(), LIMITS)
        assert decision.proposal.symbol == "AAPL"
        assert decision.limits == LIMITS


class TestBuyRejections:
    def test_missing_stop_loss(self) -> None:
        decision = evaluate_trade(buy(stop=None), make_portfolio(), LIMITS)
        assert not decision.approved
        assert [v.rule for v in decision.violations] == ["missing_stop_loss"]

    def test_stop_optional_when_not_required(self) -> None:
        limits = RiskLimits(require_stop_loss=False)
        decision = evaluate_trade(buy(stop=None), make_portfolio(), limits)
        assert decision.approved

    def test_invalid_stop_above_entry(self) -> None:
        decision = evaluate_trade(buy(stop="105"), make_portfolio(), LIMITS)
        assert "invalid_stop" in [v.rule for v in decision.violations]

    def test_risk_per_trade_exceeded(self) -> None:
        # 200 shares with $10 stop distance = $2000 = 2% of 100k > 1% limit.
        decision = evaluate_trade(buy("200", "100", "90"), make_portfolio(), LIMITS)
        assert "risk_per_trade" in [v.rule for v in decision.violations]
        assert decision.metrics["risk_pct"] == 2.0

    def test_insufficient_cash(self) -> None:
        decision = evaluate_trade(buy("2000", "100", "99.9"), make_portfolio("100000"), LIMITS)
        assert "insufficient_cash" in [v.rule for v in decision.violations]

    def test_position_concentration_counts_existing_holding(self) -> None:
        # Holding 15k of AAPL; buying 10k more -> 25% of ~115k equity > 20%.
        portfolio = make_portfolio("100000", (position(quantity="150"),))
        decision = evaluate_trade(buy("100", "100", "99.5"), portfolio, LIMITS)
        assert "position_concentration" in [v.rule for v in decision.violations]

    def test_total_exposure(self) -> None:
        limits = RiskLimits(max_total_exposure_pct=50.0, max_position_pct=100.0)
        # Existing 40k exposure + 20k new on ~140k equity -> ~42.8% ok;
        # on 100k cash+40k pos equity=140k... make it exceed: use 80k new.
        portfolio = make_portfolio("100000", (position("MSFT", "400", "100"),))
        decision = evaluate_trade(buy("800", "100", "99.9"), portfolio, limits)
        assert "total_exposure" in [v.rule for v in decision.violations]

    def test_max_positions_blocks_new_symbol_only(self) -> None:
        limits = RiskLimits(max_positions=2)
        held = (position("MSFT"), position("GOOG"))
        portfolio = make_portfolio("100000", held)

        new_symbol = evaluate_trade(buy(symbol="AAPL"), portfolio, limits)
        assert "max_positions" in [v.rule for v in new_symbol.violations]

        add_to_existing = evaluate_trade(buy(symbol="MSFT"), portfolio, limits)
        assert "max_positions" not in [v.rule for v in add_to_existing.violations]

    def test_daily_loss_halt(self) -> None:
        portfolio = make_portfolio("100000", realized_pnl_today="-4000")
        decision = evaluate_trade(buy(), portfolio, LIMITS)
        assert "daily_loss_halt" in [v.rule for v in decision.violations]

    def test_drawdown_halt(self) -> None:
        portfolio = make_portfolio("75000", equity_peak="100000")  # 25% drawdown
        decision = evaluate_trade(buy(), portfolio, LIMITS)
        assert "drawdown_halt" in [v.rule for v in decision.violations]

    def test_multiple_violations_all_reported(self) -> None:
        portfolio = make_portfolio("1000", equity_peak="100000")  # deep drawdown, no cash
        decision = evaluate_trade(buy("500", "100", stop=None), portfolio, LIMITS)
        rules = {v.rule for v in decision.violations}
        assert {"drawdown_halt", "insufficient_cash", "missing_stop_loss"} <= rules


class TestSells:
    def test_sell_of_held_position_is_approved(self) -> None:
        portfolio = make_portfolio("0", (position(quantity="50"),))
        decision = evaluate_trade(sell("50"), portfolio, LIMITS)
        assert decision.approved

    def test_sell_is_allowed_even_when_halted(self) -> None:
        """A halted account must still be able to reduce risk."""
        portfolio = make_portfolio(
            "0", (position(quantity="50"),), equity_peak="100000"
        )  # massive drawdown
        decision = evaluate_trade(sell("50"), portfolio, LIMITS)
        assert decision.approved

    def test_short_selling_rejected(self) -> None:
        decision = evaluate_trade(sell("10", "TSLA"), make_portfolio(), LIMITS)
        assert [v.rule for v in decision.violations] == ["no_position"]

    def test_oversell_rejected(self) -> None:
        portfolio = make_portfolio("0", (position(quantity="50"),))
        decision = evaluate_trade(sell("60"), portfolio, LIMITS)
        assert [v.rule for v in decision.violations] == ["oversell"]


class TestSolvency:
    def test_zero_equity_rejects_everything(self) -> None:
        decision = evaluate_trade(buy(), make_portfolio("0"), LIMITS)
        assert [v.rule for v in decision.violations] == ["solvency"]
