"""Built-in example strategies.

These serve as CLI demos, test fixtures, and reference shapes for the Strategy
Generation agent's output. They are illustrations of the schema, not
recommendations.
"""

from __future__ import annotations

from typing import Final

from atp.domain.models.strategy import (
    Comparison,
    Condition,
    Operand,
    OperandKind,
    StrategyDefinition,
)


def _indicator(kind: OperandKind, period: int) -> Operand:
    return Operand(kind=kind, period=period)


def _const(value: float) -> Operand:
    return Operand(kind=OperandKind.CONSTANT, value=value)


_CLOSE = Operand(kind=OperandKind.CLOSE)

PRESETS: Final[dict[str, StrategyDefinition]] = {
    "ema_cross": StrategyDefinition(
        name="ema_cross",
        description="Trend following: enter when EMA(20) crosses above EMA(50), "
        "exit on the reverse cross, 3-ATR stop loss.",
        entry=(
            Condition(
                left=_indicator(OperandKind.EMA, 20),
                comparison=Comparison.CROSSES_ABOVE,
                right=_indicator(OperandKind.EMA, 50),
            ),
        ),
        exit=(
            Condition(
                left=_indicator(OperandKind.EMA, 20),
                comparison=Comparison.CROSSES_BELOW,
                right=_indicator(OperandKind.EMA, 50),
            ),
        ),
        stop_loss_atr=3.0,
    ),
    "rsi_reversion": StrategyDefinition(
        name="rsi_reversion",
        description="Mean reversion: enter oversold (RSI < 30), exit when RSI "
        "recovers above 55, 2-ATR stop loss.",
        entry=(
            Condition(
                left=_indicator(OperandKind.RSI, 14),
                comparison=Comparison.LT,
                right=_const(30.0),
            ),
        ),
        exit=(
            Condition(
                left=_indicator(OperandKind.RSI, 14),
                comparison=Comparison.GT,
                right=_const(55.0),
            ),
        ),
        stop_loss_atr=2.0,
    ),
    "sma_breakout": StrategyDefinition(
        name="sma_breakout",
        description="Enter when price crosses above SMA(50), exit on the "
        "reverse cross or a 4-ATR take profit.",
        entry=(
            Condition(
                left=_CLOSE,
                comparison=Comparison.CROSSES_ABOVE,
                right=_indicator(OperandKind.SMA, 50),
            ),
        ),
        exit=(
            Condition(
                left=_CLOSE,
                comparison=Comparison.CROSSES_BELOW,
                right=_indicator(OperandKind.SMA, 50),
            ),
        ),
        take_profit_atr=4.0,
    ),
}
