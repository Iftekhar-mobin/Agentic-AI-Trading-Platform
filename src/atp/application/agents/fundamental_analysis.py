"""Fundamental Analysis agent.

Division of labor is identical to the technical agent: fixed threshold rules
classify every metric, and the LLM interprets those classifications. It is told
explicitly that the thresholds are broad-market conventions, because the whole
point of the interpretation layer is knowing when a 40x multiple is expensive
and when it is normal for the sector.
"""

from __future__ import annotations

import json
from collections import Counter

import structlog

from atp.application.agents.evidence import warn_on_unknown_sources
from atp.domain.errors import InsufficientDataError
from atp.domain.models.analysis import SignalDirection
from atp.domain.models.fundamentals import (
    MIN_FUNDAMENTAL_READINGS,
    FundamentalAssessment,
    FundamentalReport,
    Fundamentals,
)
from atp.domain.ports.llm import LLMClient
from atp.domain.services import fundamental_rules

log = structlog.get_logger()

AGENT_NAME = "fundamental_analysis"

SYSTEM_PROMPT = """\
You are the Fundamental Analysis Agent of an AI trading platform. Traders use \
your output as decision support, never as guaranteed predictions.

You receive a company's fundamental metrics, each already classified as \
bullish, bearish, or neutral by fixed threshold rules. Interpret them into an \
overall fundamental assessment.

Rules:
- Never invent or recompute numbers; cite only the values provided.
- Every evidence item's "source" must be the name of one provided reading.
- The classification thresholds are broad-market conventions, not sector- \
adjusted. Say so when they mislead: a high multiple can be reasonable for a \
fast-growing company, and a low one can signal a value trap. Use the sector \
and industry fields for that judgment.
- Weigh conflicting readings honestly; mixed signals must lower confidence.
- Confidence is calibrated 0-1: 0.5 means genuinely mixed evidence, values \
above 0.8 require agreement across valuation, profitability, growth, and \
financial health.
- Fewer metrics means less certainty. Coverage gaps must lower confidence.
- Invalidation conditions must be concrete and observable, referencing the \
metrics provided (e.g. "revenue growth turning negative in the next report").
- This is a fundamentals-only view; do not speculate about price action, \
technical levels, or news.
"""


class FundamentalAnalysisAgent:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def analyze(self, fundamentals: Fundamentals) -> FundamentalReport:
        readings = fundamental_rules.classify(fundamentals)
        if len(readings) < MIN_FUNDAMENTAL_READINGS:
            msg = (
                f"{fundamentals.symbol}: {len(readings)} fundamental metrics available, "
                f"{MIN_FUNDAMENTAL_READINGS} required for an assessment"
            )
            raise InsufficientDataError(msg)

        counts = Counter(reading.direction for reading in readings)
        signal_counts = {direction: counts.get(direction, 0) for direction in SignalDirection}

        payload = {
            "symbol": fundamentals.symbol,
            "as_of": fundamentals.as_of.isoformat(),
            "currency": fundamentals.currency,
            "sector": fundamentals.sector,
            "industry": fundamentals.industry,
            "signal_counts": {d.value: n for d, n in signal_counts.items()},
            "readings": [reading.model_dump(mode="json") for reading in readings],
        }
        assessment = await self._llm.generate_structured(
            FundamentalAssessment,
            system=SYSTEM_PROMPT,
            prompt=json.dumps(payload, sort_keys=True),
        )

        warn_on_unknown_sources(
            AGENT_NAME,
            fundamentals.symbol,
            assessment.evidence,
            {reading.name for reading in readings},
        )
        log.info(
            "fundamental_analysis.completed",
            symbol=fundamentals.symbol,
            direction=assessment.direction.value,
            confidence=assessment.confidence,
            readings=len(readings),
        )
        return FundamentalReport(
            symbol=fundamentals.symbol,
            as_of=fundamentals.as_of,
            fundamentals=fundamentals,
            readings=readings,
            signal_counts=signal_counts,
            assessment=assessment,
        )
