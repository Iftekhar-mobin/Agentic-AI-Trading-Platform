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
    timeframes = "/".join(interval.value for interval in state.intervals)
    lines = [f"{state.symbol} ({timeframes}) analysis."]

    if (technical := state.technical_report) is not None:
        per_timeframe = ", ".join(
            f"{frame.interval.value}={frame.direction.value}" for frame in technical.timeframes
        )
        lines.append(
            f"Technical: {technical.assessment.direction.value} at confidence "
            f"{technical.assessment.confidence:.2f}, close {technical.latest_close:.2f}, "
            f"timeframes {per_timeframe}. {technical.assessment.timeframe_alignment} "
            f"{technical.assessment.reasoning}"
        )

    if (patterns := state.chart_pattern_report) is not None:
        detected = [
            f"{frame.interval.value}:{pattern.kind.value}"
            f"{'' if pattern.confirmed else ' (forming)'}"
            for frame in patterns.timeframes
            for pattern in frame.patterns
        ]
        lines.append(
            f"Chart patterns: {patterns.assessment.direction.value} at confidence "
            f"{patterns.assessment.confidence:.2f}. "
            f"Detected: {', '.join(detected) if detected else 'none'}. "
            f"{patterns.assessment.reasoning}"
        )

    if (research := state.market_research_report) is not None:
        lines.append(
            f"Market research vs {research.benchmark}: "
            f"{research.assessment.direction.value} at confidence "
            f"{research.assessment.confidence:.2f}. {research.assessment.reasoning}"
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
