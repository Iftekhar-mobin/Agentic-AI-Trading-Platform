"""Structured logging via structlog.

Development renders human-readable colored console lines; staging/production
emit one JSON object per line for log aggregation. Events carry ISO-8601 UTC
timestamps, and ``structlog.contextvars`` is merged into every event so request
or workflow correlation IDs bind once and appear on all subsequent log lines.
"""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(*, level: str = "INFO", json_output: bool = False) -> None:
    numeric_level = logging.getLevelNamesMapping().get(level.upper(), logging.INFO)

    # Third-party libraries log through stdlib; keep them at the same threshold.
    logging.basicConfig(level=numeric_level, stream=sys.stdout, format="%(message)s")

    processors: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]
    if json_output:
        processors += [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=False,
    )
