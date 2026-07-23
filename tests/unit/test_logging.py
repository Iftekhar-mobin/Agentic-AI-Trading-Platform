"""Unit tests for structured logging configuration."""

from __future__ import annotations

import json

import pytest
import structlog

from atp.infrastructure.observability import configure_logging


def test_json_output_emits_parseable_json(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(level="INFO", json_output=True)
    structlog.get_logger().info("test.event", ticker="AAPL", score=0.87)

    line = capsys.readouterr().out.strip()
    event = json.loads(line)
    assert event["event"] == "test.event"
    assert event["ticker"] == "AAPL"
    assert event["score"] == 0.87
    assert event["level"] == "info"
    assert "timestamp" in event


def test_level_filtering_suppresses_lower_levels(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(level="WARNING", json_output=True)
    log = structlog.get_logger()
    log.info("should.not.appear")
    log.warning("should.appear")

    out = capsys.readouterr().out
    assert "should.not.appear" not in out
    assert "should.appear" in out


def test_contextvars_are_merged_into_events(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(level="INFO", json_output=True)
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(correlation_id="abc-123")
    try:
        structlog.get_logger().info("test.event")
    finally:
        structlog.contextvars.clear_contextvars()

    event = json.loads(capsys.readouterr().out.strip())
    assert event["correlation_id"] == "abc-123"


def test_console_output_is_human_readable(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(level="INFO", json_output=False)
    structlog.get_logger().info("test.event", ticker="AAPL")

    out = capsys.readouterr().out
    assert "test.event" in out
    assert "AAPL" in out
    with pytest.raises(json.JSONDecodeError):
        json.loads(out.strip())
