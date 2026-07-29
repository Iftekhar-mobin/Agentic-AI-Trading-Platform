"""Technical Analysis agent.

Division of labor:
- The IndicatorEngine computes and classifies every number (deterministic).
- The LLM interprets the readings into an assessment with the explainability
  envelope; it receives only precomputed values and may not invent numbers.
"""

from __future__ import annotations

import json
from collections import Counter

import structlog

from atp.application.agents.evidence import warn_on_unknown_sources
from atp.domain.errors import InsufficientHistoryError
from atp.domain.models.analysis import (
    SignalDirection,
    TechnicalAssessment,
    TechnicalReport,
)
from atp.domain.models.market import PriceHistory
from atp.domain.ports.indicators import IndicatorEngine
from atp.domain.ports.llm import LLMClient

log = structlog.get_logger()

AGENT_NAME = "technical_analysis"

SYSTEM_PROMPT = """\
You are the Technical Analysis Agent of an AI trading platform. Traders use \
your output as decision support, never as guaranteed predictions.

You receive precomputed, deterministic indicator readings for one symbol. \
Interpret them into an overall technical assessment.

Rules:
- Never invent or recompute numbers; cite only the values provided.
- Every evidence item's "source" must be the name of one provided reading.
- Weigh conflicting signals honestly; mixed signals must lower confidence.
- Confidence is calibrated 0-1: 0.5 means genuinely mixed evidence, values \
above 0.8 require strong multi-indicator agreement.
- Invalidation conditions must be concrete and observable, using price levels \
from the provided values where possible (e.g. "a daily close below the SMA(50) \
at 187.40").
- This is a single-symbol technical view only; do not speculate about news, \
fundamentals, or macro conditions.
"""


class TechnicalAnalysisAgent:
    def __init__(self, llm: LLMClient, indicators: IndicatorEngine) -> None:
        self._llm = llm
        self._indicators = indicators

    async def analyze(self, history: PriceHistory) -> TechnicalReport:
        latest = history.latest
        if latest is None:
            msg = f"{history.symbol}: no bars available to analyze"
            raise InsufficientHistoryError(msg)

        readings = self._indicators.compute_readings(history)
        counts = Counter(reading.direction for reading in readings)
        signal_counts = {direction: counts.get(direction, 0) for direction in SignalDirection}

        payload = {
            "symbol": history.symbol,
            "interval": history.interval.value,
            "as_of": latest.timestamp.isoformat(),
            "latest_close": latest.close,
            "signal_counts": {d.value: n for d, n in signal_counts.items()},
            "readings": [reading.model_dump(mode="json") for reading in readings],
        }
        assessment = await self._llm.generate_structured(
            TechnicalAssessment,
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
            "technical_analysis.completed",
            symbol=history.symbol,
            interval=history.interval.value,
            direction=assessment.direction.value,
            confidence=assessment.confidence,
            readings=len(readings),
        )
        return TechnicalReport(
            symbol=history.symbol,
            interval=history.interval,
            as_of=latest.timestamp,
            latest_close=latest.close,
            readings=readings,
            signal_counts=signal_counts,
            assessment=assessment,
        )
