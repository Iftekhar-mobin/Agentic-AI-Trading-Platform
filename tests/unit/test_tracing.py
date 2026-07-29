"""Tests for OpenTelemetry configuration and log/trace correlation."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import structlog

from atp.infrastructure.config import ObservabilitySettings
from atp.infrastructure.observability import bind_trace_context, configure_tracing
from atp.infrastructure.observability.tracing import reset_for_tests


@pytest.fixture(autouse=True)
def _reset_tracing() -> Iterator[None]:
    reset_for_tests()
    yield
    reset_for_tests()
    structlog.contextvars.clear_contextvars()


def test_tracing_stays_off_without_an_endpoint() -> None:
    """An exporter with nowhere to send spans is pure overhead."""
    assert configure_tracing(ObservabilitySettings()) is False


def test_binding_trace_context_is_a_noop_when_tracing_is_off() -> None:
    structlog.contextvars.clear_contextvars()
    bind_trace_context()
    assert "trace_id" not in structlog.contextvars.get_contextvars()


def test_tracing_configures_once_and_binds_ids() -> None:
    # Sample nothing: span ids are still issued, so correlation is exercised,
    # but no batch is ever handed to the exporter and the test needs no collector.
    settings = ObservabilitySettings(
        otlp_endpoint="http://localhost:4318/v1/traces",
        service_name="atp-test",
        trace_sample_ratio=0.0,
    )
    assert configure_tracing(settings) is True
    # Repeat calls must not stack a second exporter onto the provider.
    assert configure_tracing(settings) is True

    from opentelemetry import trace

    tracer = trace.get_tracer("test")
    structlog.contextvars.clear_contextvars()
    with tracer.start_as_current_span("unit-test-span"):
        bind_trace_context()
        bound = structlog.contextvars.get_contextvars()

    assert len(bound["trace_id"]) == 32
    assert len(bound["span_id"]) == 16
    assert int(bound["trace_id"], 16) != 0


def test_sample_ratio_is_validated() -> None:
    with pytest.raises(ValueError, match="trace_sample_ratio"):
        ObservabilitySettings(trace_sample_ratio=1.5)
