"""``atp`` CLI entry point.

Commands:
    atp status                     -- show configuration wiring (smoke check)
    atp sync AAPL -i 1d            -- pull bars from the provider into storage
    atp bars AAPL -i 1d -n 10      -- show stored bars
"""

from __future__ import annotations

import argparse
import asyncio

import structlog

from atp import __version__
from atp.composition import Container
from atp.domain.models.market import BarInterval
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

    return parser


def main() -> None:
    args = _build_parser().parse_args()
    settings = get_settings()
    configure_logging(level=settings.log_level, json_output=settings.use_json_logs)

    if args.command == "status":
        _status(settings)
        return
    asyncio.run(_run(args, settings))


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
        await init_db(container.engine)
        interval = BarInterval(args.interval)

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


if __name__ == "__main__":
    main()
