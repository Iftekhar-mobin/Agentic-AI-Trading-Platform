"""Typed application settings loaded from environment variables / .env.

All configuration enters the system through this module. Nothing else reads
``os.environ`` directly, so every setting is discoverable, typed, and validated
in one place.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache

from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class TradingMode(StrEnum):
    PAPER = "paper"
    LIVE = "live"


class DatabaseSettings(BaseModel):
    """PostgreSQL / TimescaleDB connection settings (env: ``ATP_DATABASE__*``)."""

    host: str = "localhost"
    port: int = 5432
    user: str = "atp"
    password: SecretStr = SecretStr("atp")
    name: str = "atp"

    @property
    def dsn(self) -> str:
        return (
            f"postgresql+asyncpg://{self.user}:{self.password.get_secret_value()}"
            f"@{self.host}:{self.port}/{self.name}"
        )


class LLMSettings(BaseModel):
    """LLM access settings (env: ``ATP_LLM__*``).

    ``anthropic_api_key`` is optional: when unset, the Anthropic SDK resolves
    credentials from the environment (``ANTHROPIC_API_KEY`` or an auth profile).
    """

    model: str = "claude-opus-4-8"
    max_tokens: int = 16000
    anthropic_api_key: SecretStr | None = None


class RedisSettings(BaseModel):
    """Redis connection settings (env: ``ATP_REDIS__*``)."""

    host: str = "localhost"
    port: int = 6379
    db: int = 0

    @property
    def url(self) -> str:
        return f"redis://{self.host}:{self.port}/{self.db}"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ATP_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: Environment = Environment.DEVELOPMENT
    log_level: str = "INFO"
    log_json: bool | None = Field(
        default=None,
        description="Force JSON log output. Defaults to JSON outside development.",
    )
    trading_mode: TradingMode = TradingMode.PAPER

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)

    @property
    def use_json_logs(self) -> bool:
        if self.log_json is not None:
            return self.log_json
        return self.environment is not Environment.DEVELOPMENT

    @model_validator(mode="after")
    def _forbid_live_trading_outside_production(self) -> Settings:
        if self.trading_mode is TradingMode.LIVE and self.environment is not Environment.PRODUCTION:
            msg = (
                "trading_mode=live is only permitted when environment=production; "
                f"got environment={self.environment.value}"
            )
            raise ValueError(msg)
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance (cached after first load)."""
    return Settings()
