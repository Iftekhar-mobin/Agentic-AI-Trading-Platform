"""Unit tests for the settings system.

Settings are constructed with ``_env_file=None`` so a developer's local ``.env``
can never influence test outcomes.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from atp.infrastructure.config import (
    ApiKey,
    Environment,
    Scope,
    SecuritySettings,
    Settings,
    TradingMode,
)


def _settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)


def _production(**overrides: Any) -> Settings:
    """Production settings need a credential; every production test supplies one."""
    api = SecuritySettings(keys=(ApiKey(name="test", key="k", scopes=(Scope.READ, Scope.TRADE)),))
    return _settings(environment="production", api=api, **overrides)


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
        settings = _production(trading_mode="live")
        assert settings.trading_mode is TradingMode.LIVE


class TestLogFormatDefaults:
    def test_development_defaults_to_console_logs(self) -> None:
        assert _settings(environment="development").use_json_logs is False

    def test_production_defaults_to_json_logs(self) -> None:
        assert _production(trading_mode="paper").use_json_logs is True

    def test_explicit_override_wins(self) -> None:
        assert _settings(environment="development", log_json=True).use_json_logs is True

    def test_secrets_are_masked_in_repr(self) -> None:
        settings = _settings()
        assert "atp" not in repr(settings.database.password)


class TestApiKeyGuard:
    def test_production_requires_a_credential(self) -> None:
        with pytest.raises(ValidationError, match="requires at least one API key"):
            _settings(environment="production")

    def test_production_starts_with_a_credential(self) -> None:
        assert _production().api.auth_required is True

    def test_development_may_run_unauthenticated(self) -> None:
        assert _settings().api.auth_required is False

    def test_scopes_are_checked_per_key(self) -> None:
        read_only = ApiKey(name="dashboard", key="k", scopes=(Scope.READ,))
        assert read_only.allows(Scope.READ)
        assert not read_only.allows(Scope.TRADE)

    def test_keys_are_masked_in_reprs(self) -> None:
        key = ApiKey(name="dashboard", key="super-secret", scopes=(Scope.READ,))
        assert "super-secret" not in repr(key)
