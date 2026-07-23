"""``atp`` CLI entry point.

Currently a smoke command that proves configuration and logging wire up
correctly; real subcommands arrive with the orchestration milestone.
"""

from __future__ import annotations

import structlog

from atp import __version__
from atp.infrastructure.config import get_settings
from atp.infrastructure.observability import configure_logging


def main() -> None:
    settings = get_settings()
    configure_logging(level=settings.log_level, json_output=settings.use_json_logs)

    log = structlog.get_logger()
    log.info(
        "atp.startup",
        version=__version__,
        environment=settings.environment.value,
        trading_mode=settings.trading_mode.value,
        database_host=settings.database.host,
        redis_host=settings.redis.host,
    )


if __name__ == "__main__":
    main()
