"""News agent.

Unlike the technical and fundamental agents there is no deterministic layer to
classify first — a headline is language, and interpreting language is the one
job an LLM should have. What stays code-enforced is traceability: the model
sees article ids and must cite them, so every theme it names can be walked back
to the article that produced it.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from datetime import UTC, datetime

import structlog

from atp.application.agents.evidence import warn_on_unknown_sources
from atp.domain.errors import InsufficientDataError
from atp.domain.models.news import NewsArticle, NewsAssessment, NewsReport
from atp.domain.ports.llm import LLMClient

log = structlog.get_logger()

AGENT_NAME = "news_analysis"

SYSTEM_PROMPT = """\
You are the News Agent of an AI trading platform. Traders use your output as \
decision support, never as guaranteed predictions.

You receive recent news articles about one symbol, newest first, each with an \
id, headline, summary, publisher and publication time. Identify what is \
actually happening and what it means for the stock.

Rules:
- Every evidence item's "source" must be the id of one provided article.
- Report only what the articles say. Do not add outside knowledge, and do not \
infer numbers that are not written in the text.
- key_themes are the distinct storylines driving coverage (e.g. "regulatory \
probe into ad practices"), not a restatement of each headline.
- Distinguish material events (earnings, guidance, M&A, litigation, \
regulatory action, leadership change) from routine coverage and opinion \
pieces. Weight material events far more heavily.
- Recency matters: a week-old downgrade is weaker evidence than this \
morning's guidance cut. Publication times are provided; use them.
- Beware double counting — many outlets rewrite one story. Repetition of the \
same event is not confirmation.
- Confidence is calibrated 0-1: thin, stale, or purely speculative coverage \
must keep it below 0.5.
- Invalidation conditions must be concrete and observable (e.g. "the company \
denies the acquisition report").
"""


class NewsAgent:
    def __init__(
        self,
        llm: LLMClient,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._llm = llm
        self._clock = clock

    async def analyze(
        self,
        symbol: str,
        articles: Sequence[NewsArticle],
        *,
        lookback_days: int,
    ) -> NewsReport:
        symbol = symbol.strip().upper()
        if not articles:
            msg = f"{symbol}: no articles in the last {lookback_days} days to analyze"
            raise InsufficientDataError(msg)

        now = self._clock()
        payload = {
            "symbol": symbol,
            "as_of": now.isoformat(),
            "lookback_days": lookback_days,
            "article_count": len(articles),
            "articles": [
                {
                    "id": article.id,
                    "title": article.title,
                    "summary": article.summary,
                    "publisher": article.publisher,
                    "published_at": article.published_at.isoformat(),
                    "age_hours": round((now - article.published_at).total_seconds() / 3600.0, 1),
                }
                for article in articles
            ],
        }
        assessment = await self._llm.generate_structured(
            NewsAssessment,
            system=SYSTEM_PROMPT,
            prompt=json.dumps(payload, sort_keys=True),
        )

        warn_on_unknown_sources(
            AGENT_NAME, symbol, assessment.evidence, {article.id for article in articles}
        )
        log.info(
            "news_analysis.completed",
            symbol=symbol,
            direction=assessment.direction.value,
            confidence=assessment.confidence,
            articles=len(articles),
            themes=len(assessment.key_themes),
        )
        return NewsReport(
            symbol=symbol,
            as_of=now,
            lookback_days=lookback_days,
            articles=tuple(articles),
            assessment=assessment,
        )
