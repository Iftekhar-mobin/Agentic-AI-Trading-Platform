"""Typed application settings loaded from environment variables / .env.

All configuration enters the system through this module. Nothing else reads
``os.environ`` directly, so every setting is discoverable, typed, and validated
in one place.
"""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from atp.domain.models.trading import RiskLimits


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


class SentimentModelName(StrEnum):
    LEXICON = "lexicon"
    FINBERT = "finbert"


class NewsSettings(BaseModel):
    """News retrieval settings (env: ``ATP_NEWS__*``).

    ``cache_ttl_seconds`` collapses the duplicate fetch the news and sentiment
    agents would otherwise make in the same fan-out; set it to 0 to disable.
    """

    lookback_days: int = Field(default=7, ge=1, le=90)
    limit: int = Field(default=20, ge=1, le=100)
    cache_ttl_seconds: int = Field(default=300, ge=0)


class SentimentSettings(BaseModel):
    """Sentiment classifier settings (env: ``ATP_SENTIMENT__*``).

    ``finbert`` needs the optional dependency group: ``uv sync --extra finbert``.
    """

    model: SentimentModelName = SentimentModelName.LEXICON
    finbert_model_name: str = "ProsusAI/finbert"
    half_life_days: float = Field(
        default=3.0,
        gt=0,
        le=90,
        description="Age at which an article's sentiment carries half weight",
    )


class Scope(StrEnum):
    """What an API key is allowed to do.

    Deliberately coarse: reading analysis and portfolio state is very different
    from spending money, and everything in between is a policy question this
    platform should not be guessing at.
    """

    READ = "read"
    TRADE = "trade"


class ApiKey(BaseModel):
    """One credential. ``name`` is for the audit log; the key itself is never logged."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    key: SecretStr
    scopes: tuple[Scope, ...] = (Scope.READ,)

    def allows(self, scope: Scope) -> bool:
        return scope in self.scopes


class SecuritySettings(BaseModel):
    """API authentication and hardening (env: ``ATP_API__*``).

    ``keys`` is JSON in the environment, e.g.::

        ATP_API__KEYS='[{"name":"dashboard","key":"...","scopes":["read"]}]'

    With no keys configured the API runs unauthenticated, which is convenient
    on a laptop and unacceptable anywhere else — ``Settings`` refuses to start
    in production without them.
    """

    keys: tuple[ApiKey, ...] = ()
    cors_origins: tuple[str, ...] = ()
    rate_limit_per_minute: int = Field(
        default=60,
        ge=0,
        le=100_000,
        description="Requests per minute per credential (0 disables the limiter)",
    )

    @property
    def auth_required(self) -> bool:
        return bool(self.keys)


class ObservabilitySettings(BaseModel):
    """Tracing settings (env: ``ATP_OBSERVABILITY__*``).

    Tracing stays off until an OTLP endpoint is configured: an exporter with
    nowhere to send spans is pure overhead.
    """

    otlp_endpoint: str | None = None
    service_name: str = "atp"
    trace_sample_ratio: float = Field(default=1.0, ge=0.0, le=1.0)


class MemoryBackend(StrEnum):
    JSON = "json"
    QDRANT = "qdrant"


class MemorySettings(BaseModel):
    """Episodic memory settings (env: ``ATP_MEMORY__*``).

    ``json`` is the offline default; ``qdrant`` needs the service from
    docker-compose. Changing ``embedding_dimensions`` changes the vector space,
    which orphans anything already stored — the adapters filter old rows out
    rather than comparing across spaces.
    """

    backend: MemoryBackend = MemoryBackend.JSON
    embedding_dimensions: int = Field(default=256, ge=8, le=4096)
    recall_limit: int = Field(default=5, ge=1, le=50)
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "atp_episodes"


class ExecutionSettings(BaseModel):
    """Execution settings (env: ``ATP_EXECUTION__*``)."""

    slippage_bps: float = Field(default=5.0, ge=0, le=100)


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
    data_dir: Path = Path("data")
    paper_starting_cash: Decimal = Field(default=Decimal("100000"), gt=0)

    # Hard risk limits (env: ATP_RISK__*). Enforced by the deterministic
    # risk engine; nothing downstream can loosen them at runtime.
    risk: RiskLimits = Field(default_factory=RiskLimits)
    execution: ExecutionSettings = Field(default_factory=ExecutionSettings)
    news: NewsSettings = Field(default_factory=NewsSettings)
    sentiment: SentimentSettings = Field(default_factory=SentimentSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    api: SecuritySettings = Field(default_factory=SecuritySettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)

    @property
    def use_json_logs(self) -> bool:
        if self.log_json is not None:
            return self.log_json
        return self.environment is not Environment.DEVELOPMENT

    @model_validator(mode="after")
    def _require_api_keys_in_production(self) -> Settings:
        """An unauthenticated API is a development convenience, never a deployment."""
        if self.environment is Environment.PRODUCTION and not self.api.keys:
            msg = (
                "environment=production requires at least one API key; "
                "set ATP_API__KEYS to a JSON list of {name, key, scopes}"
            )
            raise ValueError(msg)
        return self

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
