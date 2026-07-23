"""Tests for the declarative strategy schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from atp.domain.models.strategy import (
    Comparison,
    Condition,
    Operand,
    OperandKind,
    StrategyDefinition,
)
from atp.domain.strategy_presets import PRESETS


class TestOperandValidation:
    def test_indicator_requires_period(self) -> None:
        with pytest.raises(ValidationError, match="require 'period'"):
            Operand(kind=OperandKind.SMA)

    def test_constant_requires_value(self) -> None:
        with pytest.raises(ValidationError, match="require 'value'"):
            Operand(kind=OperandKind.CONSTANT)

    def test_price_kind_forbids_period_and_value(self) -> None:
        with pytest.raises(ValidationError, match="neither"):
            Operand(kind=OperandKind.CLOSE, period=14)
        with pytest.raises(ValidationError, match="neither"):
            Operand(kind=OperandKind.MACD_LINE, value=1.0)

    def test_labels(self) -> None:
        assert Operand(kind=OperandKind.EMA, period=20).label == "ema_20"
        assert Operand(kind=OperandKind.CLOSE).label == "close"
        assert Operand(kind=OperandKind.CONSTANT, value=30.0).label == "const_30.0"


class TestStrategyDefinition:
    def test_requires_entry_and_exit(self) -> None:
        condition = Condition(
            left=Operand(kind=OperandKind.CLOSE),
            comparison=Comparison.GT,
            right=Operand(kind=OperandKind.CONSTANT, value=1.0),
        )
        with pytest.raises(ValidationError):
            StrategyDefinition(name="x", entry=(), exit=(condition,))
        with pytest.raises(ValidationError):
            StrategyDefinition(name="x", entry=(condition,), exit=())

    def test_warmup_is_longest_indicator_period(self) -> None:
        assert PRESETS["ema_cross"].warmup_bars == 50
        assert PRESETS["rsi_reversion"].warmup_bars == 14
        assert PRESETS["sma_breakout"].warmup_bars == 50

    def test_macd_warmup(self) -> None:
        strategy = StrategyDefinition(
            name="macd",
            entry=(
                Condition(
                    left=Operand(kind=OperandKind.MACD_LINE),
                    comparison=Comparison.CROSSES_ABOVE,
                    right=Operand(kind=OperandKind.MACD_SIGNAL),
                ),
            ),
            exit=(
                Condition(
                    left=Operand(kind=OperandKind.MACD_LINE),
                    comparison=Comparison.CROSSES_BELOW,
                    right=Operand(kind=OperandKind.MACD_SIGNAL),
                ),
            ),
        )
        assert strategy.warmup_bars == 35

    def test_json_round_trip(self) -> None:
        """Strategies must survive serialization: they are stored and LLM-generated."""
        for preset in PRESETS.values():
            restored = StrategyDefinition.model_validate_json(preset.model_dump_json())
            assert restored == preset

    def test_describe(self) -> None:
        condition = PRESETS["ema_cross"].entry[0]
        assert condition.describe() == "ema_20 crosses_above ema_50"
