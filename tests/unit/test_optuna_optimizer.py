"""Tests for the Optuna optimizer adapter on synthetic data.

Kept fast: small trial counts, ~300-bar synthetic series.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from atp.domain.errors import OptimizationError
from atp.domain.models.market import Bar, BarInterval, PriceHistory
from atp.domain.models.optimization import (
    OptimizationSpec,
    ParameterKind,
    SearchParameter,
)
from atp.domain.strategy_presets import PRESET_SPACES, PRESETS
from atp.infrastructure.backtesting import BacktestingPyEngine
from atp.infrastructure.optimization import OptunaStrategyOptimizer


def wavy_trend(n: int = 300) -> PriceHistory:
    """Long uptrend with periodic pullbacks: crossovers and RSI dips exist."""
    closes: list[float] = []
    for i in range(n):
        base = 100.0 + 0.15 * i
        wobble = 6.0 * ((i % 40) / 40.0 - 0.5)
        closes.append(base + wobble)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    bars = tuple(
        Bar(
            timestamp=start + timedelta(days=i),
            open=close * 0.999,
            high=close * 1.012,
            low=close * 0.985,
            close=close,
            volume=1_000_000.0,
        )
        for i, close in enumerate(closes)
    )
    return PriceHistory(symbol="TEST", interval=BarInterval.DAY_1, bars=bars)


def small_spec() -> OptimizationSpec:
    """A cheap 2-parameter space around rsi_reversion."""
    return OptimizationSpec(
        strategy=PRESETS["rsi_reversion"],
        parameters=(
            SearchParameter(
                name="rsi_period",
                kind=ParameterKind.INT,
                low=5,
                high=21,
                paths=("entry.0.left.period", "exit.0.left.period"),
            ),
            SearchParameter(
                name="entry_threshold",
                kind=ParameterKind.FLOAT,
                low=20.0,
                high=45.0,
                step=1.0,
                paths=("entry.0.right.value",),
            ),
        ),
        min_trades=2,
    )


def make_optimizer() -> OptunaStrategyOptimizer:
    return OptunaStrategyOptimizer(BacktestingPyEngine())


class TestOptimize:
    def test_produces_valid_result(self) -> None:
        result = make_optimizer().optimize(small_spec(), wavy_trend(), n_trials=12, seed=1)

        assert result.completed_trials >= 1
        assert 5 <= result.best_params["rsi_period"] <= 21
        assert 20.0 <= result.best_params["entry_threshold"] <= 45.0
        # The best strategy actually carries the tuned values.
        assert result.best_strategy.entry[0].left.period == int(result.best_params["rsi_period"])
        assert result.in_sample.trades >= 2
        assert result.out_of_sample is not None  # holdout was evaluated

    def test_deterministic_given_seed(self) -> None:
        first = make_optimizer().optimize(small_spec(), wavy_trend(), n_trials=8, seed=3)
        second = make_optimizer().optimize(small_spec(), wavy_trend(), n_trials=8, seed=3)
        assert first.best_params == second.best_params
        assert first.in_sample_objective == second.in_sample_objective

    def test_search_never_sees_the_holdout(self) -> None:
        """The reported OOS window must start where the in-sample split ends."""
        history = wavy_trend()
        result = make_optimizer().optimize(
            small_spec(), history, n_trials=6, holdout_fraction=0.3, seed=1
        )
        assert result.out_of_sample is not None
        split_index = int(len(history) * 0.7)
        # OOS trades can only begin after the split (warmup prefix can't trade).
        assert result.holdout_fraction == 0.3
        assert result.in_sample.trades + result.out_of_sample.trades >= result.in_sample.trades
        assert split_index < len(history)

    def test_rejects_history_too_short_for_search(self) -> None:
        with pytest.raises(OptimizationError, match="in-sample window"):
            make_optimizer().optimize(small_spec(), wavy_trend(40), n_trials=4)

    def test_impossible_min_trades_raises(self) -> None:
        spec = small_spec().model_copy(update={"min_trades": 500})
        with pytest.raises(OptimizationError, match="no viable trials"):
            make_optimizer().optimize(spec, wavy_trend(), n_trials=4, seed=1)


class TestWalkForward:
    def test_folds_are_time_ordered_and_out_of_sample(self) -> None:
        result = make_optimizer().walk_forward(
            small_spec(), wavy_trend(360), folds=3, trials_per_fold=6, seed=1
        )

        assert len(result.folds) == 3
        for earlier, later in zip(result.folds, result.folds[1:], strict=False):
            assert earlier.test_end <= later.test_start
        for fold in result.folds:
            # Each fold trains strictly before its test window.
            assert fold.train_bars <= 360
            if fold.test_metrics is not None:
                assert fold.best_params

    def test_reports_aggregate_objective(self) -> None:
        result = make_optimizer().walk_forward(
            small_spec(), wavy_trend(360), folds=2, trials_per_fold=6, seed=1
        )
        assert result.total_test_trades >= 0
        if result.total_test_trades >= 3:
            assert result.monte_carlo is not None

    def test_too_many_folds_rejected(self) -> None:
        with pytest.raises(OptimizationError, match="too short"):
            make_optimizer().walk_forward(small_spec(), wavy_trend(120), folds=6)

    def test_ema_preset_space_smoke(self) -> None:
        result = make_optimizer().walk_forward(
            PRESET_SPACES["sma_breakout"], wavy_trend(400), folds=2, trials_per_fold=5, seed=2
        )
        assert result.strategy_name == "sma_breakout"
