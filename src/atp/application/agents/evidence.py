"""Shared post-check on agent explainability envelopes.

Every agent cites its evidence by source name. A model that invents a source
has stepped outside the data it was given, which is exactly the failure mode
the envelope exists to catch — so it is logged with the offending names rather
than passing silently.

This is a warning, not a rejection: a fabricated citation on an otherwise sound
assessment should be visible in the audit trail, not a workflow failure.
"""

from __future__ import annotations

from collections.abc import Iterable

import structlog

from atp.domain.models.explainability import Evidence

log = structlog.get_logger()


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
