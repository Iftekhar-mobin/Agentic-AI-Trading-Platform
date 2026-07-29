"""Renders the analysis artifacts into the text the memory layer works on.

This digest is doing double duty, which is why it lives in one place: it is
both the recall *query* (embedded and matched against past episodes) and the
episode *summary* that gets stored for future runs to find. Keeping one
renderer means today's stored text and tomorrow's query text are drawn from the
same vocabulary, which is most of what makes lexical or semantic recall work.

It is deliberately dense and factual — directions, confidences, named themes —
rather than prose. Embeddings match on content words, and adjectives from a
model's narrative voice would only add noise.
"""

from __future__ import annotations

from atp.application.orchestration.state import TradingState


def render_situation(state: TradingState) -> str:
    """Build a compact, factual digest of everything the analysis pool produced."""
    lines = [f"{state.symbol} ({state.interval.value}) analysis."]

    if (technical := state.technical_report) is not None:
        counts = ", ".join(
            f"{direction.value}={count}" for direction, count in technical.signal_counts.items()
        )
        lines.append(
            f"Technical: {technical.assessment.direction.value} at confidence "
            f"{technical.assessment.confidence:.2f}, close {technical.latest_close:.2f}, "
            f"indicator signals {counts}. {technical.assessment.reasoning}"
        )

    if (fundamental := state.fundamental_report) is not None:
        sector = fundamental.fundamentals.sector or "unknown sector"
        lines.append(
            f"Fundamentals: {fundamental.assessment.direction.value} at confidence "
            f"{fundamental.assessment.confidence:.2f} ({sector}). "
            f"{fundamental.assessment.reasoning}"
        )

    if (news := state.news_report) is not None:
        themes = "; ".join(news.assessment.key_themes)
        lines.append(
            f"News: {news.assessment.direction.value} at confidence "
            f"{news.assessment.confidence:.2f} over {len(news.articles)} articles. "
            f"Themes: {themes}. {news.assessment.reasoning}"
        )

    if (sentiment := state.sentiment_report) is not None:
        summary = sentiment.summary
        lines.append(
            f"Sentiment: {sentiment.assessment.direction.value} at confidence "
            f"{sentiment.assessment.confidence:.2f}, weighted polarity "
            f"{summary.weighted_polarity:+.2f} over {summary.article_count} articles "
            f"({sentiment.model_name}). {sentiment.assessment.reasoning}"
        )

    return "\n".join(lines)
