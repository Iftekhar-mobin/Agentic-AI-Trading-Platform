"""Declarative trading strategy definitions.

Strategies are *data*, not code: entry/exit rules are trees of typed operands
and comparisons. That makes them serializable (stored in Postgres, returned by
the Strategy Generation agent as structured LLM output, mutated by the
optimizer) and safely executable — the backtest engine interprets them; no
generated code ever runs.

Semantics (long-only for now; short support is a later, deliberate addition):
- ``entry``: ALL conditions must hold to open a position.
- ``exit``: ANY condition closes the position.
- ``stop_loss_atr`` / ``take_profit_atr``: optional brackets as ATR multiples.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

_MACD_WARMUP = 35  # slow EMA(26) + signal EMA(9)


class OperandKind(StrEnum):
    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    CLOSE = "close"
    VOLUME = "volume"
    CONSTANT = "constant"
    SMA = "sma"
    EMA = "ema"
    RSI = "rsi"
    ATR = "atr"
    MACD_LINE = "macd_line"
    MACD_SIGNAL = "macd_signal"
    BOLLINGER_UPPER = "bollinger_upper"
    BOLLINGER_MIDDLE = "bollinger_middle"
    BOLLINGER_LOWER = "bollinger_lower"


PERIOD_KINDS = frozenset(
    {
        OperandKind.SMA,
        OperandKind.EMA,
        OperandKind.RSI,
        OperandKind.ATR,
        OperandKind.BOLLINGER_UPPER,
        OperandKind.BOLLINGER_MIDDLE,
        OperandKind.BOLLINGER_LOWER,
    }
)


class Comparison(StrEnum):
    GT = "gt"
    LT = "lt"
    CROSSES_ABOVE = "crosses_above"
    CROSSES_BELOW = "crosses_below"


class Operand(BaseModel):
    """A value in a condition: a price field, an indicator, or a constant."""

    model_config = ConfigDict(frozen=True)

    kind: OperandKind
    period: int | None = Field(default=None, ge=2, le=500)
    value: float | None = None

    @model_validator(mode="after")
    def _check_kind_fields(self) -> Operand:
        # Strict validation matters: these models will be produced by an LLM,
        # and a malformed operand must fail loudly at parse time.
        if self.kind is OperandKind.CONSTANT:
            if self.value is None or self.period is not None:
                msg = "constant operands require 'value' and forbid 'period'"
                raise ValueError(msg)
        elif self.kind in PERIOD_KINDS:
            if self.period is None or self.value is not None:
                msg = f"'{self.kind.value}' operands require 'period' and forbid 'value'"
                raise ValueError(msg)
        elif self.period is not None or self.value is not None:
            msg = f"'{self.kind.value}' operands take neither 'period' nor 'value'"
            raise ValueError(msg)
        return self

    @property
    def label(self) -> str:
        if self.kind is OperandKind.CONSTANT:
            return f"const_{self.value}"
        if self.period is not None:
            return f"{self.kind.value}_{self.period}"
        return self.kind.value


class Condition(BaseModel):
    model_config = ConfigDict(frozen=True)

    left: Operand
    comparison: Comparison
    right: Operand

    def describe(self) -> str:
        return f"{self.left.label} {self.comparison.value} {self.right.label}"


class StrategyDefinition(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1, max_length=64)
    description: str = ""
    entry: tuple[Condition, ...] = Field(min_length=1, description="ALL must hold to enter")
    exit: tuple[Condition, ...] = Field(min_length=1, description="ANY closes the position")
    stop_loss_atr: float | None = Field(default=None, gt=0, le=20)
    take_profit_atr: float | None = Field(default=None, gt=0, le=20)
    atr_period: int = Field(default=14, ge=2, le=100)

    @property
    def warmup_bars(self) -> int:
        """Bars needed before every referenced indicator has a defined value."""
        periods = [
            operand.period
            for condition in self.entry + self.exit
            for operand in (condition.left, condition.right)
            if operand.period is not None
        ]
        if any(
            operand.kind in {OperandKind.MACD_LINE, OperandKind.MACD_SIGNAL}
            for condition in self.entry + self.exit
            for operand in (condition.left, condition.right)
        ):
            periods.append(_MACD_WARMUP)
        if self.stop_loss_atr is not None or self.take_profit_atr is not None:
            periods.append(self.atr_period)
        return max(periods, default=1)

    @property
    def min_history_bars(self) -> int:
        """Minimum bars for a meaningful backtest: warmup plus room to trade."""
        return self.warmup_bars + 10
