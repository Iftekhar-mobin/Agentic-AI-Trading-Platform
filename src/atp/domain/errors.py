"""Domain-level errors.

Infrastructure adapters translate vendor exceptions into these, so the
application layer never depends on SQLAlchemy/Anthropic/etc. exception types.
"""

from __future__ import annotations


class DomainError(Exception):
    """Base class for all domain errors."""


class InsufficientDataError(DomainError):
    """Not enough input data to form a defensible view."""


class InsufficientHistoryError(InsufficientDataError):
    """Not enough bars to compute the requested analysis."""


class RepositoryUnavailableError(DomainError):
    """The persistence backend cannot be reached (e.g. database down)."""


class LLMGenerationError(DomainError):
    """The LLM failed to produce a usable structured response."""


class ModelUnavailableError(DomainError):
    """A local ML model could not be loaded (missing optional dependency, bad weights)."""


class OptimizationError(DomainError):
    """Optimization could not produce a valid result (no viable trials, bad splits)."""


class ExecutionError(DomainError):
    """Order execution failed or would corrupt portfolio state."""
