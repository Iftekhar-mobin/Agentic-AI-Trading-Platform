"""Strategy optimization models.

A search space is declarative, like strategies themselves: each parameter
names a value range and the dotted paths inside the StrategyDefinition it
binds to. One parameter may bind several paths (e.g. the same EMA period used
in both entry and exit rules), keeping tuned strategies coherent.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from atp.domain.models.backtest import BacktestMetrics
from atp.domain.models.strategy import StrategyDefinition


class ParameterKind(StrEnum):
    INT = "int"
    FLOAT = "float"


class SearchParameter(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1, max_length=64)
    kind: ParameterKind
    low: float
    high: float
    step: float | None = Field(default=None, gt=0)
    paths: tuple[str, ...] = Field(
        min_length=1,
        description="Dotted paths into StrategyDefinition, e.g. 'entry.0.left.period'",
    )

    @model_validator(mode="after")
    def _check_bounds(self) -> SearchParameter:
        if self.low >= self.high:
            msg = f"parameter '{self.name}': low must be < high"
            raise ValueError(msg)
        return self

    def midpoint(self) -> float:
        middle = (self.low + self.high) / 2.0
        return float(int(middle)) if self.kind is ParameterKind.INT else middle


class ObjectiveMetric(StrEnum):
    SHARPE = "sharpe"
    SORTINO = "sortino"
    TOTAL_RETURN = "total_return"
    PROFIT_FACTOR = "profit_factor"
    EXPECTANCY = "expectancy"


def objective_value(metrics: BacktestMetrics, objective: ObjectiveMetric) -> float | None:
    match objective:
        case ObjectiveMetric.SHARPE:
            return metrics.sharpe
        case ObjectiveMetric.SORTINO:
            return metrics.sortino
        case ObjectiveMetric.TOTAL_RETURN:
            return metrics.total_return_pct
        case ObjectiveMetric.PROFIT_FACTOR:
            return metrics.profit_factor
        case ObjectiveMetric.EXPECTANCY:
            return metrics.expectancy_pct


class OptimizationSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    strategy: StrategyDefinition
    parameters: tuple[SearchParameter, ...] = Field(min_length=1)
    objective: ObjectiveMetric = ObjectiveMetric.SHARPE
    min_trades: int = Field(
        default=5,
        ge=1,
        description="Overfitting control: trials with fewer trades are discarded as noise",
    )

    @model_validator(mode="after")
    def _check_names_unique_and_paths_valid(self) -> OptimizationSpec:
        names = [parameter.name for parameter in self.parameters]
        if len(names) != len(set(names)):
            msg = "parameter names must be unique"
            raise ValueError(msg)
        # Fail at construction, not mid-search: applying midpoints exercises
        # every path against the actual strategy shape.
        from atp.domain.services.strategy_params import apply_parameters

        try:
            apply_parameters(
                self.strategy,
                self.parameters,
                {parameter.name: parameter.midpoint() for parameter in self.parameters},
            )
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            msg = f"invalid parameter binding: {exc}"
            raise ValueError(msg) from exc
        return self


class MonteCarloSummary(BaseModel):
    """Bootstrap resampling of trade returns: how fragile is the result?"""

    model_config = ConfigDict(frozen=True)

    simulations: int
    seed: int
    trades_resampled: int
    return_p5_pct: float
    return_p50_pct: float
    return_p95_pct: float
    max_drawdown_p50_pct: float
    max_drawdown_p95_pct: float  # the bad tail (more negative than p50)
    probability_of_loss: float


class OptimizationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    strategy_name: str
    objective: ObjectiveMetric
    requested_trials: int
    completed_trials: int
    best_params: dict[str, float]
    best_strategy: StrategyDefinition
    holdout_fraction: float
    in_sample: BacktestMetrics
    in_sample_objective: float
    out_of_sample: BacktestMetrics | None
    out_of_sample_objective: float | None
    monte_carlo: MonteCarloSummary | None

    @property
    def degradation_pct(self) -> float | None:
        """How much of the in-sample objective survives out of sample.

        Positive means decay (the usual); large values flag overfitting.
        """
        if self.out_of_sample_objective is None or self.in_sample_objective == 0:
            return None
        loss = self.in_sample_objective - self.out_of_sample_objective
        return 100.0 * loss / abs(self.in_sample_objective)


class WalkForwardFold(BaseModel):
    model_config = ConfigDict(frozen=True)

    fold: int
    train_bars: int
    test_start: datetime
    test_end: datetime
    best_params: dict[str, float]
    objective: float | None
    test_metrics: BacktestMetrics | None


class WalkForwardResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    strategy_name: str
    objective: ObjectiveMetric
    folds: tuple[WalkForwardFold, ...]
    mean_test_objective: float | None
    total_test_trades: int
    monte_carlo: MonteCarloSummary | None
