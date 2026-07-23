"""Built-in example strategies.

These serve as CLI demos, test fixtures, and reference shapes for the Strategy
Generation agent's output. They are illustrations of the schema, not
recommendations.
"""

from __future__ import annotations

from typing import Final

from atp.domain.models.optimization import (
    OptimizationSpec,
    ParameterKind,
    SearchParameter,
)
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

# Search spaces for the presets. Note how one parameter binds multiple paths
# (entry AND exit) so tuned strategies stay coherent, and how the fast/slow
# EMA ranges are disjoint so fast < slow holds by construction.
PRESET_SPACES: Final[dict[str, OptimizationSpec]] = {
    "ema_cross": OptimizationSpec(
        strategy=PRESETS["ema_cross"],
        parameters=(
            SearchParameter(
                name="fast_period",
                kind=ParameterKind.INT,
                low=5,
                high=40,
                paths=("entry.0.left.period", "exit.0.left.period"),
            ),
            SearchParameter(
                name="slow_period",
                kind=ParameterKind.INT,
                low=45,
                high=150,
                paths=("entry.0.right.period", "exit.0.right.period"),
            ),
            SearchParameter(
                name="stop_loss_atr",
                kind=ParameterKind.FLOAT,
                low=1.0,
                high=6.0,
                step=0.5,
                paths=("stop_loss_atr",),
            ),
        ),
    ),
    "rsi_reversion": OptimizationSpec(
        strategy=PRESETS["rsi_reversion"],
        parameters=(
            SearchParameter(
                name="rsi_period",
                kind=ParameterKind.INT,
                low=5,
                high=30,
                paths=("entry.0.left.period", "exit.0.left.period"),
            ),
            SearchParameter(
                name="entry_threshold",
                kind=ParameterKind.FLOAT,
                low=15.0,
                high=40.0,
                step=1.0,
                paths=("entry.0.right.value",),
            ),
            SearchParameter(
                name="exit_threshold",
                kind=ParameterKind.FLOAT,
                low=45.0,
                high=75.0,
                step=1.0,
                paths=("exit.0.right.value",),
            ),
            SearchParameter(
                name="stop_loss_atr",
                kind=ParameterKind.FLOAT,
                low=1.0,
                high=5.0,
                step=0.5,
                paths=("stop_loss_atr",),
            ),
        ),
    ),
    "sma_breakout": OptimizationSpec(
        strategy=PRESETS["sma_breakout"],
        parameters=(
            SearchParameter(
                name="sma_period",
                kind=ParameterKind.INT,
                low=20,
                high=120,
                paths=("entry.0.right.period", "exit.0.right.period"),
            ),
            SearchParameter(
                name="take_profit_atr",
                kind=ParameterKind.FLOAT,
                low=2.0,
                high=8.0,
                step=0.5,
                paths=("take_profit_atr",),
            ),
        ),
    ),
}
