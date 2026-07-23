"""Position sizing math. Pure, deterministic, Decimal in/out.

Share counts are whole numbers (US equities); fractional-share support is a
deliberate later change, not an accident of rounding.
"""

from __future__ import annotations

from decimal import ROUND_DOWN, Decimal


def size_by_risk(
    equity: Decimal,
    risk_pct: float,
    entry_price: Decimal,
    stop_loss: Decimal,
) -> Decimal:
    """Shares such that hitting the stop loses ~``risk_pct`` percent of equity.

    The standard ATR/stop-based size: risk budget divided by per-share risk.
    """
    if equity <= 0:
        msg = "equity must be positive"
        raise ValueError(msg)
    if stop_loss >= entry_price:
        msg = f"stop loss {stop_loss} must be below entry {entry_price} for a long"
        raise ValueError(msg)
    risk_budget = equity * Decimal(str(risk_pct)) / 100
    per_share_risk = entry_price - stop_loss
    return (risk_budget / per_share_risk).to_integral_value(rounding=ROUND_DOWN)


def size_by_fraction(equity: Decimal, fraction_pct: float, price: Decimal) -> Decimal:
    """Shares worth ``fraction_pct`` percent of equity at ``price``."""
    if equity <= 0 or price <= 0:
        msg = "equity and price must be positive"
        raise ValueError(msg)
    budget = equity * Decimal(str(fraction_pct)) / 100
    return (budget / price).to_integral_value(rounding=ROUND_DOWN)


def max_affordable(cash: Decimal, price: Decimal) -> Decimal:
    """Upper bound on shares purchasable with available cash."""
    if price <= 0:
        msg = "price must be positive"
        raise ValueError(msg)
    return (cash / price).to_integral_value(rounding=ROUND_DOWN)


def kelly_fraction(win_rate: float, payoff_ratio: float) -> float:
    """Raw Kelly fraction f* = W - (1 - W) / R, clamped to [0, 1].

    ``payoff_ratio`` is average win divided by average loss. Callers should
    scale down (half-Kelly is conventional) — full Kelly assumes the input
    statistics are exact, which backtest statistics never are.
    """
    if not 0.0 <= win_rate <= 1.0:
        msg = "win_rate must be in [0, 1]"
        raise ValueError(msg)
    if payoff_ratio <= 0:
        msg = "payoff_ratio must be positive"
        raise ValueError(msg)
    fraction = win_rate - (1.0 - win_rate) / payoff_ratio
    return min(max(fraction, 0.0), 1.0)
