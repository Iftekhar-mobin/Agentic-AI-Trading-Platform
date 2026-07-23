"""Monte Carlo bootstrap over trade returns.

Resamples the observed trade sequence (with replacement) many times and
compounds each resampled sequence into a terminal return and max drawdown.
The spread of outcomes shows how much of a backtest's result is sequencing
luck — a robustness check, not a forecast. Deterministic given a seed.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from atp.domain.models.optimization import MonteCarloSummary

MIN_TRADES = 3


def monte_carlo_trades(
    trade_returns_pct: Sequence[float],
    *,
    simulations: int = 1_000,
    seed: int = 42,
) -> MonteCarloSummary | None:
    """Returns None when there are too few trades to resample meaningfully."""
    if len(trade_returns_pct) < MIN_TRADES:
        return None

    rng = random.Random(seed)
    totals: list[float] = []
    drawdowns: list[float] = []
    for _ in range(simulations):
        equity = 1.0
        peak = 1.0
        worst_drawdown = 0.0
        for _ in trade_returns_pct:
            equity *= 1.0 + rng.choice(trade_returns_pct) / 100.0
            peak = max(peak, equity)
            worst_drawdown = min(worst_drawdown, equity / peak - 1.0)
        totals.append((equity - 1.0) * 100.0)
        drawdowns.append(worst_drawdown * 100.0)

    totals.sort()
    drawdowns.sort()  # ascending: most negative (worst) first
    return MonteCarloSummary(
        simulations=simulations,
        seed=seed,
        trades_resampled=len(trade_returns_pct),
        return_p5_pct=_percentile(totals, 0.05),
        return_p50_pct=_percentile(totals, 0.50),
        return_p95_pct=_percentile(totals, 0.95),
        max_drawdown_p50_pct=_percentile(drawdowns, 0.50),
        max_drawdown_p95_pct=_percentile(drawdowns, 0.05),  # bad tail
        probability_of_loss=sum(1 for total in totals if total < 0) / simulations,
    )


def _percentile(sorted_values: list[float], quantile: float) -> float:
    index = round(quantile * (len(sorted_values) - 1))
    return sorted_values[index]
