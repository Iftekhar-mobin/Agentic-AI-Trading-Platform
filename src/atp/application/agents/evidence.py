"""Shared post-check on agent explainability envelopes.

Every agent cites its evidence by source name. A model that invents a source
has stepped outside the data it was given, which is exactly the failure mode
the envelope exists to catch — so it is logged with the offending names rather
than passing silently.

This is a warning, not a rejection: a fabricated citation on an otherwise sound
assessment should be visible in the audit trail, not a workflow failure.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import structlog

from atp.domain.models.explainability import Evidence

log = structlog.get_logger()


def external_evidence_sources(context: Mapping[str, Any] | None) -> set[str]:
    """Citable ``bot:`` source names for caller-supplied context.

    Agents are told to cite anything drawn from ``external_context`` as
    ``bot:<field>``. Without registering those names here every such citation
    would be reported as fabricated, and the warning would stop meaning
    anything. Nested keys are exposed both bare and dotted, since a model may
    reasonably cite either ``bot:trend_gate`` or ``bot:trend_gate.bias_direction``.
    """
    if not context:
        return set()
    sources: set[str] = set()
    for key, value in context.items():
        sources.add(f"bot:{key}")
        if isinstance(value, Mapping):
            sources.update(f"bot:{key}.{nested}" for nested in value)
    return sources


def warn_on_unknown_sources(
    agent: str,
    symbol: str,
    evidence: Iterable[Evidence],
    known_sources: set[str],
) -> tuple[str, ...]:
    """Log and return evidence sources that were not among the agent's inputs."""
    unknown = tuple(item.source for item in evidence if item.source not in known_sources)
    if unknown:
        log.warning(
            "agent.unknown_evidence_sources",
            agent=agent,
            symbol=symbol,
            sources=list(unknown),
        )
    return unknown
