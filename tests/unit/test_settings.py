"""Unit tests for the settings system.

Settings are constructed with ``_env_file=None`` so a developer's local ``.env``
can never influence test outcomes.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from atp.infrastructure.config import Environment, Settings, TradingMode


def _settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)


class TestDefaults:
    def test_defaults_are_safe(self) -> None:
        settings = _settings()
        assert settings.environment is Environment.DEVELOPMENT
        assert settings.trading_mode is TradingMode.PAPER
        assert settings.log_level == "INFO"

    def test_database_dsn_composition(self) -> None:
        settings = _settings()
        assert settings.database.dsn == "postgresql+asyncpg://atp:atp@localhost:5432/atp"

    def test_redis_url_composition(self) -> None:
        settings = _settings()
        assert settings.redis.url == "redis://localhost:6379/0"


class TestEnvironmentOverrides:
    def test_reads_prefixed_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ATP_ENVIRONMENT", "staging")
        monkeypatch.setenv("ATP_LOG_LEVEL", "DEBUG")
        settings = _settings()
        assert settings.environment is Environment.STAGING
        assert settings.log_level == "DEBUG"

    def test_reads_nested_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ATP_DATABASE__HOST", "db.internal")
        monkeypatch.setenv("ATP_DATABASE__PORT", "5433")
        settings = _settings()
        assert settings.database.host == "db.internal"
        assert settings.database.port == 5433


class TestLiveTradingGuard:
    def test_live_trading_rejected_outside_production(self) -> None:
        with pytest.raises(ValidationError, match="environment=production"):
            _settings(trading_mode="live", environment="development")

    def test_live_trading_allowed_in_production(self) -> None:
        settings = _settings(trading_mode="live", environment="production")
        assert settings.trading_mode is TradingMode.LIVE


class TestLogFormatDefaults:
    def test_development_defaults_to_console_logs(self) -> None:
        assert _settings(environment="development").use_json_logs is False

    def test_production_defaults_to_json_logs(self) -> None:
        assert _settings(environment="production", trading_mode="paper").use_json_logs is True

    def test_explicit_override_wins(self) -> None:
        assert _settings(environment="development", log_json=True).use_json_logs is True

    def test_secrets_are_masked_in_repr(self) -> None:
        settings = _settings()
        assert "atp" not in repr(settings.database.password)
