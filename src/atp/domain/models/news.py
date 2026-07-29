"""News domain models.

An article is evidence, not a conclusion: the news agent's job is to name the
themes and catalysts a trader would otherwise have to read fifty headlines to
find, with every claim traceable to an article id.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from atp.domain.models.analysis import SignalDirection
from atp.domain.models.explainability import Explanation


class NewsArticle(BaseModel):
    """A single normalized news item. Immutable value object."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    summary: str | None = None
    publisher: str | None = None
    url: str | None = None
    published_at: datetime
    symbols: tuple[str, ...] = ()

    @field_validator("published_at")
    @classmethod
    def _normalize_to_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            msg = "article timestamps must be timezone-aware"
            raise ValueError(msg)
        return value.astimezone(UTC)

    @property
    def text(self) -> str:
        """Title plus summary — the unit of text handed to a sentiment model."""
        return f"{self.title}. {self.summary}" if self.summary else self.title


class NewsAssessment(Explanation):
    """The LLM's read of the news flow (evidence sources are article ids)."""

    direction: SignalDirection
    key_themes: tuple[str, ...] = Field(min_length=1)


class NewsReport(BaseModel):
    """Full output of the News agent."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    as_of: datetime
    lookback_days: int = Field(gt=0)
    articles: tuple[NewsArticle, ...]
    assessment: NewsAssessment
