"""Domain-level errors.

Infrastructure adapters translate vendor exceptions into these, so the
application layer never depends on SQLAlchemy/Anthropic/etc. exception types.
"""

from __future__ import annotations


class DomainError(Exception):
    """Base class for all domain errors."""


class InsufficientHistoryError(DomainError):
    """Not enough bars to compute the requested analysis."""


class RepositoryUnavailableError(DomainError):
    """The persistence backend cannot be reached (e.g. database down)."""


class LLMGenerationError(DomainError):
    """The LLM failed to produce a usable structured response."""
