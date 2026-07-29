"""OpenTelemetry tracing setup.

Off unless ``ATP_OBSERVABILITY__OTLP_ENDPOINT`` is set — an exporter with
nowhere to send spans costs allocation and buys nothing. When it is set, spans
go out over OTLP/HTTP to whatever collector the deployment runs.

The piece that makes traces and logs usable together is ``bind_trace_context``:
it copies the active span's trace and span ids into ``structlog``'s
contextvars, so every log line carries them alongside the request id. Jumping
from a slow span in a trace UI to the exact log lines it produced is then a
text search, not an archaeology exercise.

Tracing is configured once per process and is a no-op on repeat calls, so tests
and reloading dev servers cannot stack exporters.
"""

from __future__ import annotations

import structlog

from atp.infrastructure.config.settings import ObservabilitySettings

log = structlog.get_logger()

_configured = False


def configure_tracing(settings: ObservabilitySettings) -> bool:
    """Set up the global tracer provider. Returns True when tracing is live."""
    global _configured
    if _configured:
        return True
    if not settings.otlp_endpoint:
        log.debug("tracing.disabled", reason="no OTLP endpoint configured")
        return False

    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

    provider = TracerProvider(
        resource=Resource.create({"service.name": settings.service_name}),
        # ParentBased: once a trace is sampled upstream, keep sampling it, or
        # traces arrive with holes where this service should be.
        sampler=ParentBased(TraceIdRatioBased(settings.trace_sample_ratio)),
    )
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.otlp_endpoint))
    )
    trace.set_tracer_provider(provider)

    _configured = True
    log.info(
        "tracing.configured",
        endpoint=settings.otlp_endpoint,
        service=settings.service_name,
        sample_ratio=settings.trace_sample_ratio,
    )
    return True


def bind_trace_context() -> None:
    """Bind the active span's ids into the log context, if tracing is on."""
    if not _configured:
        return
    from opentelemetry import trace

    span = trace.get_current_span()
    context = span.get_span_context()
    if not context.is_valid:
        return
    structlog.contextvars.bind_contextvars(
        trace_id=format(context.trace_id, "032x"),
        span_id=format(context.span_id, "016x"),
    )


def instrument_app(app: object) -> None:
    """Attach FastAPI/ASGI instrumentation. No-op when tracing is disabled."""
    if not _configured:
        return
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    FastAPIInstrumentor.instrument_app(app)  # type: ignore[arg-type]
    log.debug("tracing.app_instrumented")


def reset_for_tests() -> None:
    """Clear the once-per-process guard. Test-only."""
    global _configured
    _configured = False
