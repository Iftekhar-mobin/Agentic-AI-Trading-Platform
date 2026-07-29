"""Sentiment agent — the FinBERT + LLM pipeline.

Three stages, deliberately separated:

1. A sentiment classifier (FinBERT in production, the lexicon model offline)
   labels each article. This is the only step that produces numbers.
2. A deterministic aggregation weights those labels by recency and derives a
   direction from fixed thresholds — reproducible without any model call.
3. The LLM interprets the aggregate: what the tone actually reflects, whether
   it is worth trusting, and what would change it.

The LLM never scores text, and the classifier never explains itself. That split
is what makes the sentiment number auditable and the narrative useful.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from datetime import UTC, datetime

import structlog

from atp.application.agents.evidence import warn_on_unknown_sources
from atp.domain.errors import InsufficientDataError
from atp.domain.models.news import NewsArticle
from atp.domain.models.sentiment import (
    ScoredArticle,
    SentimentAssessment,
    SentimentReport,
)
from atp.domain.ports.llm import LLMClient
from atp.domain.ports.sentiment import SentimentModel
from atp.domain.services import sentiment_aggregation

log = structlog.get_logger()

AGENT_NAME = "sentiment_analysis"

SYSTEM_PROMPT = """\
You are the Sentiment Agent of an AI trading platform. Traders use your output \
as decision support, never as guaranteed predictions.

You receive news articles that a financial sentiment classifier has already \
labelled, plus a deterministic aggregate of those labels. The aggregate's \
weighted_polarity runs from -1 (uniformly negative) to +1 (uniformly \
positive) and weights recent articles more heavily.

Rules:
- Never invent or recompute scores; the classifier's labels and the aggregate \
are the only sentiment numbers that exist. You may disagree with a label, but \
say so explicitly and explain why rather than silently substituting your own.
- Every evidence item's "source" must be the id of one provided article.
- Your direction should normally match the aggregate's direction. Departing \
from it requires an explicit reason in your reasoning (for example: the \
classifier read a routine headline as negative, or one material article \
outweighs a dozen trivial positive ones).
- Sentiment is a crowd measure, not a forecast. Extremely one-sided sentiment \
is often a contrarian signal near turning points; note that when it applies.
- Confidence is calibrated 0-1. Few articles, sentiment near zero, or a \
split distribution must keep it below 0.5.
- Invalidation conditions must be concrete and observable.
"""


class SentimentAgent:
    def __init__(
        self,
        llm: LLMClient,
        model: SentimentModel,
        *,
        half_life_days: float = sentiment_aggregation.DEFAULT_HALF_LIFE_DAYS,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._llm = llm
        self._model = model
        self._half_life_days = half_life_days
        self._clock = clock

    async def analyze(self, symbol: str, articles: Sequence[NewsArticle]) -> SentimentReport:
        symbol = symbol.strip().upper()
        if not articles:
            msg = f"{symbol}: no articles available to score"
            raise InsufficientDataError(msg)

        scores = await self._model.score([article.text for article in articles])
        if len(scores) != len(articles):
            msg = (
                f"{symbol}: sentiment model '{self._model.name}' returned {len(scores)} scores "
                f"for {len(articles)} articles"
            )
            raise InsufficientDataError(msg)

        scored = tuple(
            ScoredArticle(article=article, sentiment=score)
            for article, score in zip(articles, scores, strict=True)
        )
        now = self._clock()
        summary = sentiment_aggregation.summarize(
            scored, now=now, half_life_days=self._half_life_days
        )

        payload = {
            "symbol": symbol,
            "as_of": now.isoformat(),
            "classifier": self._model.name,
            "summary": summary.model_dump(mode="json"),
            "articles": [
                {
                    "id": item.article.id,
                    "title": item.article.title,
                    "published_at": item.article.published_at.isoformat(),
                    "label": item.sentiment.label.value,
                    "confidence": item.sentiment.confidence,
                    "polarity": round(item.sentiment.polarity, 4),
                }
                for item in scored
            ],
        }
        assessment = await self._llm.generate_structured(
            SentimentAssessment,
            system=SYSTEM_PROMPT,
            prompt=json.dumps(payload, sort_keys=True),
        )

        warn_on_unknown_sources(
            AGENT_NAME, symbol, assessment.evidence, {item.article.id for item in scored}
        )
        if assessment.direction is not summary.direction:
            # Not an error - the prompt permits it with a stated reason - but it
            # is the kind of divergence a reviewer should be able to find later.
            log.info(
                "sentiment_analysis.direction_override",
                symbol=symbol,
                aggregate=summary.direction.value,
                assessed=assessment.direction.value,
            )
        log.info(
            "sentiment_analysis.completed",
            symbol=symbol,
            classifier=self._model.name,
            direction=assessment.direction.value,
            confidence=assessment.confidence,
            weighted_polarity=summary.weighted_polarity,
            articles=len(scored),
        )
        return SentimentReport(
            symbol=symbol,
            as_of=now,
            model_name=self._model.name,
            scored_articles=scored,
            summary=summary,
            assessment=assessment,
        )
