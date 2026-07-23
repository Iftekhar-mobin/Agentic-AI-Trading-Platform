"""Optuna adapter for the StrategyOptimizer port.

Overfitting controls, applied in layers:
- The search only ever sees the in-sample window; the holdout (or each
  walk-forward test fold) is evaluated exactly once, with the chosen params.
- Trials with fewer than ``spec.min_trades`` trades are pruned as noise.
- Out-of-sample windows are prefixed with warmup bars so indicators are
  defined from the window start, but warmup bars cannot generate entries.
- Monte Carlo resampling of out-of-sample trades quantifies sequencing luck.
- Results always report in-sample AND out-of-sample objectives side by side;
  the degradation between them is the overfitting signal.
"""

from __future__ import annotations

import optuna
import structlog
from optuna.samplers import TPESampler
from optuna.trial import TrialState

from atp.domain.errors import DomainError, OptimizationError
from atp.domain.models.backtest import BacktestResult
from atp.domain.models.market import PriceHistory
from atp.domain.models.optimization import (
    MonteCarloSummary,
    OptimizationResult,
    OptimizationSpec,
    ParameterKind,
    WalkForwardFold,
    WalkForwardResult,
    objective_value,
)
from atp.domain.models.strategy import StrategyDefinition
from atp.domain.ports.backtesting import BacktestEngine
from atp.domain.services import apply_parameters, monte_carlo_trades

log = structlog.get_logger()

optuna.logging.set_verbosity(optuna.logging.WARNING)

_MIN_TEST_TRADE_BARS = 10


class OptunaStrategyOptimizer:
    def __init__(self, engine: BacktestEngine) -> None:
        self._engine = engine

    def optimize(
        self,
        spec: OptimizationSpec,
        history: PriceHistory,
        *,
        n_trials: int = 50,
        holdout_fraction: float = 0.3,
        initial_cash: float = 10_000.0,
        commission: float = 0.001,
        seed: int = 42,
    ) -> OptimizationResult:
        if not 0.0 < holdout_fraction < 0.9:
            msg = f"holdout_fraction must be in (0, 0.9), got {holdout_fraction}"
            raise OptimizationError(msg)

        warmup = self._max_warmup(spec)
        split = int(len(history) * (1.0 - holdout_fraction))
        in_sample = _slice(history, 0, split)
        if len(in_sample) < warmup + _MIN_TEST_TRADE_BARS:
            msg = (
                f"in-sample window has {len(in_sample)} bars; needs at least "
                f"{warmup + _MIN_TEST_TRADE_BARS} for the widest parameter values"
            )
            raise OptimizationError(msg)

        best_params, completed = self._search(
            spec, in_sample, n_trials, seed, initial_cash, commission
        )
        best_strategy = apply_parameters(spec.strategy, spec.parameters, best_params)

        in_sample_run = self._engine.run(
            best_strategy, in_sample, initial_cash=initial_cash, commission=commission
        )
        in_sample_objective = objective_value(in_sample_run.metrics, spec.objective)
        if in_sample_objective is None:  # pragma: no cover - search guarantees it
            msg = "best trial lost its objective on re-evaluation"
            raise OptimizationError(msg)

        out_run = self._evaluate_window(
            best_strategy, history, split - warmup, len(history), initial_cash, commission
        )
        monte_carlo = _monte_carlo_of(out_run, seed)

        result = OptimizationResult(
            strategy_name=spec.strategy.name,
            objective=spec.objective,
            requested_trials=n_trials,
            completed_trials=completed,
            best_params=best_params,
            best_strategy=best_strategy,
            holdout_fraction=holdout_fraction,
            in_sample=in_sample_run.metrics,
            in_sample_objective=in_sample_objective,
            out_of_sample=out_run.metrics if out_run else None,
            out_of_sample_objective=(
                objective_value(out_run.metrics, spec.objective) if out_run else None
            ),
            monte_carlo=monte_carlo,
        )
        log.info(
            "optimization.completed",
            strategy=spec.strategy.name,
            trials=completed,
            best_params=best_params,
            in_sample_objective=round(in_sample_objective, 4),
            out_of_sample_objective=result.out_of_sample_objective,
        )
        return result

    def walk_forward(
        self,
        spec: OptimizationSpec,
        history: PriceHistory,
        *,
        folds: int = 4,
        trials_per_fold: int = 25,
        initial_cash: float = 10_000.0,
        commission: float = 0.001,
        seed: int = 42,
    ) -> WalkForwardResult:
        if folds < 2:
            msg = "walk-forward needs at least 2 folds"
            raise OptimizationError(msg)

        warmup = self._max_warmup(spec)
        total = len(history)
        test_region_start = total // 2
        test_length = (total - test_region_start) // folds
        if test_length < warmup // 2 + _MIN_TEST_TRADE_BARS:
            msg = (
                f"history too short for {folds} walk-forward folds "
                f"(test windows of {test_length} bars)"
            )
            raise OptimizationError(msg)

        fold_results: list[WalkForwardFold] = []
        test_objectives: list[float] = []
        all_test_returns: list[float] = []

        for fold in range(folds):
            test_start = test_region_start + fold * test_length
            test_end = total if fold == folds - 1 else test_start + test_length
            train = _slice(history, 0, test_start)

            try:
                best_params, _ = self._search(
                    spec, train, trials_per_fold, seed + fold, initial_cash, commission
                )
            except OptimizationError:
                log.warning("walk_forward.fold_failed", fold=fold, train_bars=len(train))
                fold_results.append(
                    WalkForwardFold(
                        fold=fold,
                        train_bars=len(train),
                        test_start=history.bars[test_start].timestamp,
                        test_end=history.bars[test_end - 1].timestamp,
                        best_params={},
                        objective=None,
                        test_metrics=None,
                    )
                )
                continue

            best_strategy = apply_parameters(spec.strategy, spec.parameters, best_params)
            test_run = self._evaluate_window(
                best_strategy, history, test_start - warmup, test_end, initial_cash, commission
            )
            test_objective = objective_value(test_run.metrics, spec.objective) if test_run else None
            if test_objective is not None:
                test_objectives.append(test_objective)
            if test_run is not None:
                all_test_returns.extend(t.return_pct for t in test_run.trade_log)

            fold_results.append(
                WalkForwardFold(
                    fold=fold,
                    train_bars=len(train),
                    test_start=history.bars[test_start].timestamp,
                    test_end=history.bars[test_end - 1].timestamp,
                    best_params=best_params,
                    objective=test_objective,
                    test_metrics=test_run.metrics if test_run else None,
                )
            )

        result = WalkForwardResult(
            strategy_name=spec.strategy.name,
            objective=spec.objective,
            folds=tuple(fold_results),
            mean_test_objective=(
                sum(test_objectives) / len(test_objectives) if test_objectives else None
            ),
            total_test_trades=len(all_test_returns),
            monte_carlo=monte_carlo_trades(all_test_returns, seed=seed),
        )
        log.info(
            "walk_forward.completed",
            strategy=spec.strategy.name,
            folds=folds,
            mean_test_objective=result.mean_test_objective,
            total_test_trades=result.total_test_trades,
        )
        return result

    def _search(
        self,
        spec: OptimizationSpec,
        history: PriceHistory,
        n_trials: int,
        seed: int,
        initial_cash: float,
        commission: float,
    ) -> tuple[dict[str, float], int]:
        """Runs the Optuna study; returns (best_params, completed_trials)."""

        def objective(trial: optuna.Trial) -> float:
            values: dict[str, float] = {}
            for parameter in spec.parameters:
                if parameter.kind is ParameterKind.INT:
                    values[parameter.name] = trial.suggest_int(
                        parameter.name,
                        int(parameter.low),
                        int(parameter.high),
                        step=int(parameter.step) if parameter.step else 1,
                    )
                else:
                    values[parameter.name] = trial.suggest_float(
                        parameter.name, parameter.low, parameter.high, step=parameter.step
                    )
            strategy = apply_parameters(spec.strategy, spec.parameters, values)
            try:
                run = self._engine.run(
                    strategy, history, initial_cash=initial_cash, commission=commission
                )
            except DomainError as exc:
                raise optuna.TrialPruned(str(exc)) from exc
            if run.metrics.trades < spec.min_trades:
                raise optuna.TrialPruned(f"only {run.metrics.trades} trades")
            value = objective_value(run.metrics, spec.objective)
            if value is None:
                raise optuna.TrialPruned(f"{spec.objective.value} undefined")
            return value

        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed))
        study.optimize(objective, n_trials=n_trials)

        completed = sum(1 for t in study.trials if t.state == TrialState.COMPLETE)
        if completed == 0:
            msg = (
                f"no viable trials out of {n_trials} for '{spec.strategy.name}' — "
                f"every sampled strategy produced fewer than {spec.min_trades} trades "
                f"or an undefined {spec.objective.value}"
            )
            raise OptimizationError(msg)
        return {k: float(v) for k, v in study.best_params.items()}, completed

    def _evaluate_window(
        self,
        strategy: StrategyDefinition,
        history: PriceHistory,
        start: int,
        end: int,
        initial_cash: float,
        commission: float,
    ) -> BacktestResult | None:
        window = _slice(history, max(0, start), end)
        if len(window) < strategy.min_history_bars:
            log.warning(
                "optimization.window_too_small",
                bars=len(window),
                needed=strategy.min_history_bars,
            )
            return None
        return self._engine.run(strategy, window, initial_cash=initial_cash, commission=commission)

    def _max_warmup(self, spec: OptimizationSpec) -> int:
        """Warmup of the strategy with every parameter at its widest value."""
        widest = apply_parameters(
            spec.strategy,
            spec.parameters,
            {parameter.name: parameter.high for parameter in spec.parameters},
        )
        return max(widest.warmup_bars, spec.strategy.warmup_bars)


def _slice(history: PriceHistory, start: int, end: int) -> PriceHistory:
    return PriceHistory(
        symbol=history.symbol, interval=history.interval, bars=history.bars[start:end]
    )


def _monte_carlo_of(run: BacktestResult | None, seed: int) -> MonteCarloSummary | None:
    if run is None:
        return None
    return monte_carlo_trades([t.return_pct for t in run.trade_log], seed=seed)
