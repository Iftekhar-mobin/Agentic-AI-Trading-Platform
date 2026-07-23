"""``atp`` CLI entry point.

Commands:
    atp status                     -- show configuration wiring (smoke check)
    atp sync AAPL -i 1d            -- pull bars from the provider into storage
    atp bars AAPL -i 1d -n 10      -- show stored bars
    atp analyze AAPL -i 1d         -- run the analysis workflow (supervisor graph)
    atp backtest AAPL -s ema_cross -- backtest a strategy preset (or -f file.json)
    atp optimize AAPL -s ema_cross -- optimize strategy parameters (add --walk-forward)
    atp serve                      -- start the HTTP API
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import timedelta
from pathlib import Path

import structlog

from atp import __version__
from atp.composition import Container
from atp.domain.errors import DomainError
from atp.domain.models.analysis import TechnicalReport
from atp.domain.models.backtest import BacktestResult
from atp.domain.models.market import BarInterval
from atp.domain.models.optimization import OptimizationResult, WalkForwardResult
from atp.domain.models.strategy import StrategyDefinition
from atp.domain.strategy_presets import PRESET_SPACES, PRESETS
from atp.infrastructure.config import Settings, get_settings
from atp.infrastructure.observability import configure_logging
from atp.infrastructure.persistence import init_db

log = structlog.get_logger()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atp", description="Agentic AI Trading Platform")
    parser.set_defaults(command="status")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("status", help="show configuration wiring")

    intervals = [interval.value for interval in BarInterval]
    sync = subparsers.add_parser("sync", help="pull bars from the market data provider")
    sync.add_argument("symbol")
    sync.add_argument("-i", "--interval", choices=intervals, default="1d")

    bars = subparsers.add_parser("bars", help="show stored bars")
    bars.add_argument("symbol")
    bars.add_argument("-i", "--interval", choices=intervals, default="1d")
    bars.add_argument("-n", "--limit", type=int, default=10)

    analyze = subparsers.add_parser("analyze", help="run the analysis workflow")
    analyze.add_argument("symbol")
    analyze.add_argument("-i", "--interval", choices=intervals, default="1d")

    backtest = subparsers.add_parser("backtest", help="backtest a strategy")
    backtest.add_argument("symbol")
    backtest.add_argument("-i", "--interval", choices=intervals, default="1d")
    source = backtest.add_mutually_exclusive_group(required=True)
    source.add_argument("-s", "--strategy", choices=sorted(PRESETS))
    source.add_argument("-f", "--file", type=Path, help="StrategyDefinition JSON file")
    backtest.add_argument("--cash", type=float, default=10_000.0)
    backtest.add_argument("--commission", type=float, default=0.001)

    optimize = subparsers.add_parser("optimize", help="optimize strategy parameters")
    optimize.add_argument("symbol")
    optimize.add_argument("-i", "--interval", choices=intervals, default="1d")
    optimize.add_argument("-s", "--strategy", choices=sorted(PRESET_SPACES), required=True)
    optimize.add_argument("-n", "--trials", type=int, default=50)
    optimize.add_argument("--walk-forward", action="store_true")
    optimize.add_argument("--folds", type=int, default=4)
    optimize.add_argument("--holdout", type=float, default=0.3)
    optimize.add_argument("--seed", type=int, default=42)
    optimize.add_argument("--years", type=float, default=5.0, help="history to fetch")

    serve = subparsers.add_parser("serve", help="start the HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    return parser


def main() -> None:
    args = _build_parser().parse_args()
    settings = get_settings()
    configure_logging(level=settings.log_level, json_output=settings.use_json_logs)

    if args.command == "status":
        _status(settings)
        return
    if args.command == "serve":
        import uvicorn

        uvicorn.run(
            "atp.interfaces.api:create_app",
            factory=True,
            host=args.host,
            port=args.port,
            log_level=settings.log_level.lower(),
        )
        return
    try:
        asyncio.run(_run(args, settings))
    except DomainError as exc:
        log.error("command.failed", command=args.command, error=str(exc))
        raise SystemExit(1) from exc


def _status(settings: Settings) -> None:
    log.info(
        "atp.status",
        version=__version__,
        environment=settings.environment.value,
        trading_mode=settings.trading_mode.value,
        database_host=settings.database.host,
        redis_host=settings.redis.host,
    )


async def _run(args: argparse.Namespace, settings: Settings) -> None:
    container = Container.build(settings)
    try:
        interval = BarInterval(args.interval)

        if args.command == "analyze":
            # Runs through the supervisor graph; no init_db, because analysis
            # falls back to the live provider when storage is down.
            state = await container.orchestrator.run(args.symbol, interval)
            for failure in state.failures:
                log.error("agent.failure", agent=failure.agent, error=failure.error)
            if state.technical_report is None:
                raise SystemExit(1)
            _print_report(state.technical_report)
            return

        if args.command == "backtest":
            strategy = (
                PRESETS[args.strategy]
                if args.strategy
                else StrategyDefinition.model_validate_json(args.file.read_text("utf-8"))
            )
            backtest_result = await container.run_backtest.execute(
                strategy,
                args.symbol,
                interval,
                initial_cash=args.cash,
                commission=args.commission,
            )
            _print_backtest(backtest_result)
            return

        if args.command == "optimize":
            spec = PRESET_SPACES[args.strategy]
            lookback = timedelta(days=365 * args.years)
            if args.walk_forward:
                wf = await container.optimize_strategy.walk_forward(
                    spec,
                    args.symbol,
                    interval,
                    folds=args.folds,
                    seed=args.seed,
                    lookback=lookback,
                )
                _print_walk_forward(wf)
            else:
                opt = await container.optimize_strategy.execute(
                    spec,
                    args.symbol,
                    interval,
                    n_trials=args.trials,
                    holdout_fraction=args.holdout,
                    seed=args.seed,
                    lookback=lookback,
                )
                _print_optimization(opt)
            return

        await init_db(container.engine)

        if args.command == "sync":
            result = await container.sync_market_data.execute(args.symbol, interval)
            log.info("sync.completed", **result.model_dump(mode="json"))
        elif args.command == "bars":
            history = await container.get_price_history.execute(
                args.symbol, interval, limit=args.limit
            )
            for bar in history.bars:
                log.info(
                    "bar",
                    symbol=history.symbol,
                    ts=bar.timestamp.isoformat(),
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume=bar.volume,
                )
            log.info("bars.total", symbol=history.symbol, count=len(history))
    finally:
        await container.aclose()


def _print_optimization(result: OptimizationResult) -> None:
    log.info(
        "optimize.result",
        strategy=result.strategy_name,
        objective=result.objective.value,
        trials=f"{result.completed_trials}/{result.requested_trials}",
        best_params=result.best_params,
    )
    log.info(
        "optimize.in_sample",
        objective=round(result.in_sample_objective, 4),
        **result.in_sample.model_dump(mode="json"),
    )
    if result.out_of_sample is not None:
        log.info(
            "optimize.out_of_sample",
            objective=result.out_of_sample_objective,
            degradation_pct=result.degradation_pct,
            **result.out_of_sample.model_dump(mode="json"),
        )
    if result.monte_carlo is not None:
        log.info("optimize.monte_carlo", **result.monte_carlo.model_dump(mode="json"))


def _print_walk_forward(result: WalkForwardResult) -> None:
    log.info(
        "walk_forward.result",
        strategy=result.strategy_name,
        objective=result.objective.value,
        mean_test_objective=result.mean_test_objective,
        total_test_trades=result.total_test_trades,
    )
    for fold in result.folds:
        log.info(
            "walk_forward.fold",
            fold=fold.fold,
            train_bars=fold.train_bars,
            test_start=fold.test_start.date().isoformat(),
            test_end=fold.test_end.date().isoformat(),
            best_params=fold.best_params,
            objective=fold.objective,
            trades=fold.test_metrics.trades if fold.test_metrics else None,
            total_return_pct=(
                round(fold.test_metrics.total_return_pct, 2) if fold.test_metrics else None
            ),
        )
    if result.monte_carlo is not None:
        log.info("walk_forward.monte_carlo", **result.monte_carlo.model_dump(mode="json"))


def _print_backtest(result: BacktestResult) -> None:
    metrics = result.metrics
    log.info(
        "backtest.run",
        strategy=result.strategy.name,
        symbol=result.symbol,
        interval=result.interval.value,
        bars=result.bars,
        start=result.start.date().isoformat(),
        end=result.end.date().isoformat(),
        initial_cash=result.initial_cash,
        final_equity=round(result.final_equity, 2),
    )
    log.info("backtest.metrics", **metrics.model_dump(mode="json"))
    for trade in result.trade_log:
        log.info(
            "trade",
            entry=trade.entry_time.date().isoformat(),
            exit=trade.exit_time.date().isoformat(),
            return_pct=round(trade.return_pct, 2),
        )


def _print_report(report: TechnicalReport) -> None:
    for reading in report.readings:
        log.info(
            "indicator",
            name=reading.name,
            direction=reading.direction.value,
            summary=reading.summary,
        )
    assessment = report.assessment
    log.info(
        "analysis.assessment",
        symbol=report.symbol,
        as_of=report.as_of.isoformat(),
        latest_close=report.latest_close,
        direction=assessment.direction.value,
        confidence=assessment.confidence,
        signal_counts={d.value: n for d, n in report.signal_counts.items()},
    )
    log.info("analysis.reasoning", text=assessment.reasoning)
    for evidence in assessment.evidence:
        log.info("analysis.evidence", source=evidence.source, statement=evidence.statement)
    for condition in assessment.invalidation_conditions:
        log.info("analysis.invalidation", condition=condition)


if __name__ == "__main__":
    main()
