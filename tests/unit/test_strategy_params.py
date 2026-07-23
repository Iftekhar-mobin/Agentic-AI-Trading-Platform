"""Tests for parameter application and search-space validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from atp.domain.models.optimization import (
    OptimizationSpec,
    ParameterKind,
    SearchParameter,
)
from atp.domain.services import apply_parameters
from atp.domain.strategy_presets import PRESET_SPACES, PRESETS


class TestApplyParameters:
    def test_applies_to_all_bound_paths(self) -> None:
        spec = PRESET_SPACES["ema_cross"]
        tuned = apply_parameters(
            spec.strategy,
            spec.parameters,
            {"fast_period": 10, "slow_period": 100, "stop_loss_atr": 2.5},
        )
        assert tuned.entry[0].left.period == 10
        assert tuned.exit[0].left.period == 10  # same param, both paths
        assert tuned.entry[0].right.period == 100
        assert tuned.exit[0].right.period == 100
        assert tuned.stop_loss_atr == 2.5

    def test_int_kind_casts(self) -> None:
        spec = PRESET_SPACES["ema_cross"]
        tuned = apply_parameters(
            spec.strategy,
            spec.parameters,
            {"fast_period": 10.7, "slow_period": 100.2, "stop_loss_atr": 2.0},
        )
        assert tuned.entry[0].left.period == 10
        assert isinstance(tuned.entry[0].left.period, int)

    def test_original_is_untouched(self) -> None:
        spec = PRESET_SPACES["ema_cross"]
        before = spec.strategy.model_copy(deep=True)
        apply_parameters(
            spec.strategy,
            spec.parameters,
            {"fast_period": 7, "slow_period": 90, "stop_loss_atr": 1.5},
        )
        assert spec.strategy == before

    def test_out_of_schema_value_fails_validation(self) -> None:
        spec = PRESET_SPACES["ema_cross"]
        with pytest.raises(ValidationError):
            apply_parameters(
                spec.strategy,
                spec.parameters,
                {"fast_period": 1, "slow_period": 100, "stop_loss_atr": 2.0},  # period < 2
            )


class TestSpecValidation:
    def test_bad_path_rejected_at_construction(self) -> None:
        with pytest.raises(ValidationError, match="invalid parameter binding"):
            OptimizationSpec(
                strategy=PRESETS["ema_cross"],
                parameters=(
                    SearchParameter(
                        name="broken",
                        kind=ParameterKind.INT,
                        low=5,
                        high=40,
                        paths=("entry.0.left.wrong_field",),
                    ),
                ),
            )

    def test_out_of_range_index_rejected(self) -> None:
        with pytest.raises(ValidationError, match="invalid parameter binding"):
            OptimizationSpec(
                strategy=PRESETS["ema_cross"],
                parameters=(
                    SearchParameter(
                        name="broken",
                        kind=ParameterKind.INT,
                        low=5,
                        high=40,
                        paths=("entry.5.left.period",),
                    ),
                ),
            )

    def test_duplicate_names_rejected(self) -> None:
        parameter = SearchParameter(
            name="p", kind=ParameterKind.INT, low=5, high=40, paths=("entry.0.left.period",)
        )
        with pytest.raises(ValidationError, match="unique"):
            OptimizationSpec(strategy=PRESETS["ema_cross"], parameters=(parameter, parameter))

    def test_low_must_be_below_high(self) -> None:
        with pytest.raises(ValidationError, match="low must be < high"):
            SearchParameter(
                name="p", kind=ParameterKind.INT, low=40, high=5, paths=("entry.0.left.period",)
            )

    def test_all_preset_spaces_are_valid(self) -> None:
        assert set(PRESET_SPACES) == set(PRESETS)
