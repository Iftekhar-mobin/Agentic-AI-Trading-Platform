"""``atp`` CLI entry point.

Commands:
    atp status                     -- show configuration wiring (smoke check)
    atp sync AAPL -i 1d            -- pull bars from the provider into storage
    atp bars AAPL -i 1d -n 10      -- show stored bars
    atp analyze AAPL -i 1d         -- run every analysis agent (supervisor graph)
    atp analyze AAPL -a news_analysis,sentiment_analysis  -- run a subset
    atp backtest AAPL -s ema_cross -- backtest a strategy preset (or -f file.json)
    atp optimize AAPL -s ema_cross -- optimize strategy parameters (add --walk-forward)
    atp portfolio                  -- show the paper portfolio with live prices
    atp risk-check AAPL --stop 300 -- size a trade and run it through the risk gate
    atp trade AAPL --stop 300      -- execute through the gate (paper broker)
    atp orders -n 20               -- show the execution audit trail
    atp memory AAPL -q "breakout"  -- recall episodes from the trade journal
    atp serve                      -- start the HTTP API
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import structlog

from atp import __version__
from atp.application.orchestration import ALL_AGENTS, TradingState, UnknownAgentError
from atp.composition import Container
from atp.domain.errors import DomainError
from atp.domain.models.analysis import TechnicalReport
from atp.domain.models.backtest import BacktestResult
from atp.domain.models.explainability import Explanation
from atp.domain.models.fundamentals import FundamentalReport
from atp.domain.models.market import BarInterval
from atp.domain.models.memory import EpisodeMatch, LearningReport
from atp.domain.models.news import NewsReport
from atp.domain.models.optimization import OptimizationResult, WalkForwardResult
from atp.domain.models.orders import Order
from atp.domain.models.sentiment import SentimentReport
from atp.domain.models.strategy import StrategyDefinition
from atp.domain.models.trading import OrderSide, Portfolio, RiskDecision
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
    analyze.add_argument(
        "-a",
        "--agents",
        default=None,
        help=(f"comma-separated subset of {','.join(ALL_AGENTS)} (default: all)"),
    )

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

    subparsers.add_parser("portfolio", help="show the paper portfolio")

    for name, help_text in [
        ("risk-check", "run a trade through the risk gate"),
        ("trade", "execute a trade through the gate (paper broker)"),
    ]:
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("symbol")
        command.add_argument("--side", choices=["buy", "sell"], default="buy")
        command.add_argument("--qty", type=Decimal, default=None)
        command.add_argument("--stop", type=Decimal, default=None, help="stop-loss price")
        command.add_argument("--take", type=Decimal, default=None, help="take-profit price")

    orders = subparsers.add_parser("orders", help="show the execution audit trail")
    orders.add_argument("-n", "--limit", type=int, default=20)

    memory = subparsers.add_parser("memory", help="recall episodes from the trade journal")
    memory.add_argument("symbol")
    memory.add_argument(
        "-q", "--query", default=None, help="free-text query (default: the symbol itself)"
    )
    memory.add_argument("-n", "--limit", type=int, default=10)

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
        if args.command == "portfolio":
            portfolio = await container.get_portfolio.execute()
            _print_portfolio(portfolio)
            return

        if args.command == "risk-check":
            decision = await container.check_trade_risk.execute(
                args.symbol,
                side=OrderSide(args.side),
                quantity=args.qty,
                stop_loss=args.stop,
                take_profit=args.take,
            )
            _print_risk_decision(decision)
            if not decision.approved:
                raise SystemExit(1)
            return

        if args.command == "trade":
            execution = await container.execute_trade.execute(
                args.symbol,
                side=OrderSide(args.side),
                quantity=args.qty,
                stop_loss=args.stop,
                take_profit=args.take,
            )
            _print_risk_decision(execution.decision)
            if execution.order is not None:
                _print_order(execution.order)
            # Journal every outcome, rejections included: "the gate refused
            # this" is a precedent worth recalling later.
            await container.journal_trade.execute(execution)
            if not execution.executed:
                raise SystemExit(1)
            portfolio = await container.portfolio_repository.load()
            _print_portfolio(portfolio)
            return

        if args.command == "memory":
            matches = await container.memory.recall(
                args.query or args.symbol, symbol=args.symbol, limit=args.limit
            )
            for match in matches:
                _print_episode(match)
            log.info("memory.total", symbol=args.symbol.upper(), count=len(matches))
            return

        if args.command == "orders":
            records = await container.order_repository.list_records(limit=args.limit)
            for record in records:
                _print_order(record.order)
            log.info("orders.total", count=len(records))
            return

        interval = BarInterval(args.interval)

        if args.command == "analyze":
            # Runs through the supervisor graph; no init_db, because analysis
            # falls back to the live provider when storage is down.
            agents = args.agents.split(",") if args.agents else None
            try:
                state = await container.orchestrator.run(args.symbol, interval, agents=agents)
            except UnknownAgentError as exc:
                log.error("analyze.unknown_agents", error=str(exc))
                raise SystemExit(1) from exc
            for failure in state.failures:
                log.error("agent.failure", agent=failure.agent, error=failure.error)
            if not state.has_report:
                raise SystemExit(1)
            _print_analysis(state)
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


def _print_portfolio(portfolio: Portfolio) -> None:
    log.info(
        "portfolio.summary",
        cash=float(portfolio.cash),
        equity=float(portfolio.equity),
        exposure_pct=round(float(portfolio.exposure_pct), 2),
        drawdown_pct=round(float(portfolio.drawdown_pct), 2),
        positions=len(portfolio.positions),
        realized_pnl_today=float(portfolio.realized_pnl_today),
    )
    for position in portfolio.positions:
        log.info(
            "position",
            symbol=position.symbol,
            quantity=float(position.quantity),
            avg_entry=float(position.avg_entry_price),
            price=float(position.current_price),
            market_value=float(position.market_value),
            unrealized_pnl=round(float(position.unrealized_pnl), 2),
            unrealized_pnl_pct=round(float(position.unrealized_pnl_pct), 2),
        )


def _print_order(order: Order) -> None:
    log.info(
        "order",
        id=order.id,
        symbol=order.symbol,
        side=order.side.value,
        quantity=float(order.quantity),
        status=order.status.value,
        fill_price=float(order.fill_price) if order.fill_price else None,
        submitted_at=order.submitted_at.isoformat(),
        strategy=order.strategy_name,
        reason=order.reason,
    )


def _print_risk_decision(decision: RiskDecision) -> None:
    proposal = decision.proposal
    log.info(
        "risk.proposal",
        symbol=proposal.symbol,
        side=proposal.side.value,
        quantity=float(proposal.quantity),
        entry_price=float(proposal.entry_price),
        stop_loss=float(proposal.stop_loss) if proposal.stop_loss else None,
        notional=round(float(proposal.notional), 2),
    )
    log.info("risk.metrics", **{k: round(v, 4) for k, v in decision.metrics.items()})
    if decision.approved:
        log.info("risk.verdict", verdict="APPROVED")
    else:
        log.error("risk.verdict", verdict="REJECTED")
        for violation in decision.violations:
            log.error("risk.violation", rule=violation.rule, detail=violation.detail)


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


def _print_explanation(prefix: str, explanation: Explanation) -> None:
    """Print the envelope every agent carries: why, on what, and what refutes it."""
    log.info(f"{prefix}.reasoning", text=explanation.reasoning)
    for evidence in explanation.evidence:
        log.info(f"{prefix}.evidence", source=evidence.source, statement=evidence.statement)
    for condition in explanation.invalidation_conditions:
        log.info(f"{prefix}.invalidation", condition=condition)


def _print_episode(match: EpisodeMatch) -> None:
    episode = match.episode
    log.info(
        "episode",
        id=episode.id,
        kind=episode.kind.value,
        occurred_at=episode.occurred_at.isoformat(),
        regime=episode.regime.label if episode.regime else None,
        similarity=round(match.score, 4),
        summary=episode.summary,
    )


def _print_learning(report: LearningReport) -> None:
    for match in report.recalled:
        _print_episode(match)
    entry = report.entry
    log.info(
        "learning.report",
        symbol=report.symbol,
        as_of=report.as_of.isoformat(),
        regime=report.regime.label if report.regime else None,
        recalled=len(report.recalled),
        confidence=entry.confidence,
    )
    log.info("learning.regime_note", text=entry.regime_note)
    for lesson in entry.lessons:
        log.info("learning.lesson", text=lesson)
    _print_explanation("learning", entry)


def _print_analysis(state: TradingState) -> None:
    if state.technical_report is not None:
        _print_report(state.technical_report)
    if state.fundamental_report is not None:
        _print_fundamentals(state.fundamental_report)
    if state.news_report is not None:
        _print_news(state.news_report)
    if state.sentiment_report is not None:
        _print_sentiment(state.sentiment_report)
    if state.learning_report is not None:
        _print_learning(state.learning_report)


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
    _print_explanation("analysis", assessment)


def _print_fundamentals(report: FundamentalReport) -> None:
    for reading in report.readings:
        log.info(
            "fundamental",
            name=reading.name,
            category=reading.category.value,
            direction=reading.direction.value,
            summary=reading.summary,
        )
    assessment = report.assessment
    log.info(
        "fundamentals.assessment",
        symbol=report.symbol,
        as_of=report.as_of.isoformat(),
        sector=report.fundamentals.sector,
        industry=report.fundamentals.industry,
        direction=assessment.direction.value,
        confidence=assessment.confidence,
        signal_counts={d.value: n for d, n in report.signal_counts.items()},
    )
    _print_explanation("fundamentals", assessment)


def _print_news(report: NewsReport) -> None:
    for article in report.articles:
        log.info(
            "article",
            id=article.id,
            published=article.published_at.isoformat(),
            publisher=article.publisher,
            title=article.title,
        )
    assessment = report.assessment
    log.info(
        "news.assessment",
        symbol=report.symbol,
        as_of=report.as_of.isoformat(),
        lookback_days=report.lookback_days,
        articles=len(report.articles),
        direction=assessment.direction.value,
        confidence=assessment.confidence,
        themes=list(assessment.key_themes),
    )
    _print_explanation("news", assessment)


def _print_sentiment(report: SentimentReport) -> None:
    for item in report.scored_articles:
        log.info(
            "sentiment.article",
            id=item.article.id,
            label=item.sentiment.label.value,
            confidence=item.sentiment.confidence,
            title=item.article.title,
        )
    summary = report.summary
    assessment = report.assessment
    log.info(
        "sentiment.summary",
        symbol=report.symbol,
        as_of=report.as_of.isoformat(),
        classifier=report.model_name,
        articles=summary.article_count,
        label_counts={label.value: n for label, n in summary.label_counts.items()},
        mean_polarity=round(summary.mean_polarity, 4),
        weighted_polarity=round(summary.weighted_polarity, 4),
        aggregate_direction=summary.direction.value,
    )
    log.info(
        "sentiment.assessment",
        symbol=report.symbol,
        direction=assessment.direction.value,
        confidence=assessment.confidence,
    )
    _print_explanation("sentiment", assessment)


if __name__ == "__main__":
    main()
