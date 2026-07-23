"""Shared test configuration."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import structlog


@pytest.fixture(autouse=True)
def _isolate_structlog_config() -> Iterator[None]:
    """Keep structlog configuration from leaking between tests."""
    yield
    structlog.reset_defaults()
