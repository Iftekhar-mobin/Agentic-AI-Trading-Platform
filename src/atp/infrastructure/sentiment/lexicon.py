"""Deterministic lexicon sentiment model — the default, offline classifier.

Why this exists alongside FinBERT: the test suite must run offline and CI must
not download a 400 MB transformer, but a stub that returns "neutral" would make
the sentiment agent untestable in any meaningful way. This adapter is a real,
if blunt, classifier built on a Loughran-McDonald-style financial word list
with simple negation handling, so its output is reproducible and explainable
line by line.

It is a baseline, not the production path: ``FinBertSentimentModel`` is what
``ATP_SENTIMENT__MODEL=finbert`` selects, and it understands context this
cannot ("cut costs" vs "cut guidance").
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from atp.domain.models.sentiment import SentimentLabel, SentimentScore

_TOKEN_RE = re.compile(r"[a-z][a-z'\-]*")

NEGATORS = frozenset(
    {
        "no",
        "not",
        "never",
        "none",
        "nor",
        "without",
        "fails",
        "failed",
        "fail",
        "failing",
        "unlikely",
        "unable",
        "lacks",
        "lack",
        "lacking",
        "cannot",
        "cant",
        "wont",
        "isnt",
        "arent",
        "didnt",
        "doesnt",
        "despite",
    }
)
"""Flip the polarity of a sentiment word appearing shortly after one of these."""

NEGATION_WINDOW = 3
"""How many tokens a negator reaches forward."""

# fmt: off - word lists stay compact; one term per line would be 200 lines of noise
POSITIVE_TERMS = frozenset(
    {
        "accelerate",
        "accelerated",
        "accelerating",
        "advance",
        "advanced",
        "approval",
        "approved",
        "award",
        "awarded",
        "beat",
        "beats",
        "best",
        "boost",
        "boosted",
        "breakthrough",
        "bullish",
        "buyback",
        "climb",
        "climbed",
        "confident",
        "expand",
        "expanded",
        "expansion",
        "exceed",
        "exceeded",
        "exceeds",
        "gain",
        "gained",
        "gains",
        "grew",
        "grow",
        "growth",
        "high",
        "higher",
        "improve",
        "improved",
        "improvement",
        "jump",
        "jumped",
        "leading",
        "milestone",
        "momentum",
        "opportunity",
        "optimistic",
        "outperform",
        "outperformed",
        "outperforming",
        "partnership",
        "positive",
        "profit",
        "profitable",
        "profits",
        "rallied",
        "rally",
        "rebound",
        "rebounded",
        "record",
        "recovery",
        "resilient",
        "rise",
        "rises",
        "rising",
        "robust",
        "rose",
        "soar",
        "soared",
        "solid",
        "strength",
        "strong",
        "stronger",
        "success",
        "successful",
        "surge",
        "surged",
        "surpass",
        "surpassed",
        "top",
        "topped",
        "tops",
        "upbeat",
        "upgrade",
        "upgraded",
        "upside",
        "win",
        "wins",
        "won",
    }
)

NEGATIVE_TERMS = frozenset(
    {
        "bankruptcy",
        "bearish",
        "breach",
        "concern",
        "concerns",
        "crash",
        "cut",
        "cuts",
        "decline",
        "declined",
        "declines",
        "default",
        "deficit",
        "delay",
        "delayed",
        "disappoint",
        "disappointed",
        "disappointing",
        "downgrade",
        "downgraded",
        "downside",
        "downturn",
        "drop",
        "dropped",
        "drops",
        "fell",
        "fraud",
        "halt",
        "halted",
        "headwind",
        "headwinds",
        "investigation",
        "lawsuit",
        "layoff",
        "layoffs",
        "loss",
        "losses",
        "lost",
        "lower",
        "miss",
        "missed",
        "misses",
        "negative",
        "penalty",
        "pessimistic",
        "plunge",
        "plunged",
        "probe",
        "recall",
        "recession",
        "resign",
        "resigned",
        "restructuring",
        "scandal",
        "selloff",
        "shortfall",
        "shrink",
        "shrank",
        "sink",
        "sank",
        "slash",
        "slashed",
        "slowdown",
        "slump",
        "slumped",
        "struggle",
        "struggled",
        "struggling",
        "sued",
        "suspend",
        "suspended",
        "tumble",
        "tumbled",
        "underperform",
        "underperformed",
        "warn",
        "warned",
        "warning",
        "weak",
        "weaker",
        "weakness",
        "worries",
        "worry",
    }
)
# fmt: on

NEUTRAL_BAND = 0.25
"""Below this net-to-total ratio the text reads as neutral rather than directional."""


class LexiconSentimentModel:
    """SentimentModel implementation over a fixed financial word list."""

    @property
    def name(self) -> str:
        return "lexicon-v1"

    async def score(self, texts: Sequence[str]) -> tuple[SentimentScore, ...]:
        # Pure CPU string work on short headlines; no thread offload needed.
        return tuple(self.score_text(text) for text in texts)

    @classmethod
    def score_text(cls, text: str) -> SentimentScore:
        tokens = _TOKEN_RE.findall(text.lower())
        positive = 0
        negative = 0
        negator_at: int | None = None

        for index, token in enumerate(tokens):
            if token in NEGATORS:
                negator_at = index
                continue
            negated = negator_at is not None and index - negator_at <= NEGATION_WINDOW
            if token in POSITIVE_TERMS:
                negative += 1 if negated else 0
                positive += 0 if negated else 1
            elif token in NEGATIVE_TERMS:
                positive += 1 if negated else 0
                negative += 0 if negated else 1

        return cls._to_score(positive, negative)

    @staticmethod
    def _to_score(positive: int, negative: int) -> SentimentScore:
        total = positive + negative
        if total == 0:
            # No sentiment-bearing words: neutral, but only weakly asserted.
            return SentimentScore(label=SentimentLabel.NEUTRAL, confidence=0.6)

        strength = abs(positive - negative) / total
        if strength < NEUTRAL_BAND:
            # Balanced positive and negative language reads as genuinely mixed.
            return SentimentScore(
                label=SentimentLabel.NEUTRAL,
                confidence=round(0.5 + 0.2 * (1.0 - strength), 4),
            )
        label = SentimentLabel.POSITIVE if positive > negative else SentimentLabel.NEGATIVE
        return SentimentScore(label=label, confidence=round(min(0.95, 0.5 + 0.45 * strength), 4))
