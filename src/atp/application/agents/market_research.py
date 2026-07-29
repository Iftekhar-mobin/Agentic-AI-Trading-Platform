"""Market Research agent.

Answers the question every single-symbol agent structurally cannot: how does
this instrument sit relative to the market it trades in? A stock up 4% on the
month looks strong until you learn the index rose 9%.

Deterministic layer: relative strength over several windows, beta and
correlation, the benchmark's own trend, and drawdown comparison — all computed
on the intersection of the two series' timestamps. The LLM interprets what that
relationship means for positioning.
"""

from __future__ import annotations

import json
from collections import Counter

import structlog

from atp.application.agents.evidence import warn_on_unknown_sources
from atp.domain.models.analysis import SignalDirection
from atp.domain.models.market import PriceHistory
from atp.domain.models.research import (
    MarketResearchAssessment,
    MarketResearchReport,
)
from atp.domain.ports.llm import LLMClient
from atp.domain.services import market_context

log = structlog.get_logger()

AGENT_NAME = "market_research"

SYSTEM_PROMPT = """\
You are the Market Research Agent of an AI trading platform. Traders use your \
output as decision support, never as guaranteed predictions.

You receive precomputed measurements of one symbol relative to a benchmark: \
relative strength over several windows, beta and correlation, the benchmark's \
own trend, and how the two have drawn down. Interpret where this instrument \
sits within the market.

Rules:
- Never invent or recompute numbers; cite only the values provided.
- Every evidence item's "source" must be the name of one provided reading.
- Your subject is the *relationship*, not the symbol on its own. Absolute \
performance is only interesting here as a comparison.
- Separate the two questions a position depends on: is the market itself a \
tailwind or a headwind, and is this instrument leading or lagging it? Say both.
- Use beta to qualify the rest. A high-beta name outperforming a rising market \
may be delivering nothing but leverage, and will give it back on the way down.
- Relative strength across different windows can disagree; that is a change in \
leadership and worth naming explicitly rather than averaging away.
- Confidence is calibrated 0-1: conflicting windows or a low correlation to \
the benchmark must lower it.
- Invalidation conditions must be concrete and observable, referencing the \
provided measurements.
- Do not speculate about news, fundamentals, or specific chart levels.
"""


class MarketResearchAgent:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def analyze(self, history: PriceHistory, benchmark: PriceHistory) -> MarketResearchReport:
        readings = market_context.analyze(history, benchmark)
        counts = Counter(reading.direction for reading in readings)
        signal_counts = {direction: counts.get(direction, 0) for direction in SignalDirection}
        aligned_bars = len(market_context.align(history, benchmark)[0])
        latest = history.bars[-1]

        payload = {
            "symbol": history.symbol,
            "benchmark": benchmark.symbol,
            "interval": history.interval.value,
            "as_of": latest.timestamp.isoformat(),
            "overlapping_bars": aligned_bars,
            "signal_counts": {direction.value: count for direction, count in signal_counts.items()},
            "readings": [reading.model_dump(mode="json") for reading in readings],
        }
        assessment = await self._llm.generate_structured(
            MarketResearchAssessment,
            system=SYSTEM_PROMPT,
            prompt=json.dumps(payload, sort_keys=True),
        )

        warn_on_unknown_sources(
            AGENT_NAME,
            history.symbol,
            assessment.evidence,
            {reading.name for reading in readings},
        )
        log.info(
            "market_research.completed",
            symbol=history.symbol,
            benchmark=benchmark.symbol,
            direction=assessment.direction.value,
            confidence=assessment.confidence,
            readings=len(readings),
            overlapping_bars=aligned_bars,
        )
        return MarketResearchReport(
            symbol=history.symbol,
            benchmark=benchmark.symbol,
            interval=history.interval,
            as_of=latest.timestamp,
            overlapping_bars=aligned_bars,
            readings=readings,
            signal_counts=signal_counts,
            assessment=assessment,
        )
