"""FinBERT adapter for the SentimentModel port.

FinBERT is a BERT model fine-tuned on financial text; it reads "shares cut
through resistance" and "company cut guidance" differently, which a word list
cannot. It is the intended production classifier.

``transformers`` and ``torch`` are an **optional** dependency group
(``uv sync --extra finbert``) and are imported lazily, for two reasons: they
add gigabytes to an image that most deployments do not need, and the test
suite plus CI must run offline. Selecting this model without the extra
installed fails loudly at construction time rather than silently degrading.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import structlog

from atp.domain.errors import ModelUnavailableError
from atp.domain.models.sentiment import SentimentLabel, SentimentScore

log = structlog.get_logger()

DEFAULT_MODEL = "ProsusAI/finbert"

_LABELS: dict[str, SentimentLabel] = {
    "positive": SentimentLabel.POSITIVE,
    "negative": SentimentLabel.NEGATIVE,
    "neutral": SentimentLabel.NEUTRAL,
    # Some fine-tunes ship generic head labels.
    "label_0": SentimentLabel.POSITIVE,
    "label_1": SentimentLabel.NEGATIVE,
    "label_2": SentimentLabel.NEUTRAL,
}


class FinBertSentimentModel:
    """SentimentModel implementation backed by a HuggingFace text-classification pipeline."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        batch_size: int = 16,
        max_length: int = 512,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._max_length = max_length
        self._pipeline: Any | None = None

    @property
    def name(self) -> str:
        return self._model_name

    async def score(self, texts: Sequence[str]) -> tuple[SentimentScore, ...]:
        if not texts:
            return ()
        # Model inference is blocking and CPU/GPU bound; keep the loop free.
        return await asyncio.to_thread(self._score_sync, list(texts))

    def _score_sync(self, texts: list[str]) -> tuple[SentimentScore, ...]:
        pipeline = self._load()
        outputs = pipeline(
            texts,
            batch_size=self._batch_size,
            truncation=True,
            max_length=self._max_length,
        )
        return tuple(self._to_score(output) for output in outputs)

    def _load(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        try:
            from transformers import pipeline as hf_pipeline
        except ImportError as exc:
            msg = (
                f"sentiment model '{self._model_name}' needs the optional 'finbert' extra - "
                "install it with `uv sync --extra finbert`, or set "
                "ATP_SENTIMENT__MODEL=lexicon"
            )
            raise ModelUnavailableError(msg) from exc

        try:
            self._pipeline = hf_pipeline(
                "text-classification",
                model=self._model_name,
                top_k=None,  # return every class so we can read the winner's probability
            )
        except Exception as exc:  # transformers raises a wide range of errors
            msg = f"could not load sentiment model '{self._model_name}': {exc}"
            raise ModelUnavailableError(msg) from exc

        log.info("sentiment.model_loaded", model=self._model_name)
        return self._pipeline

    @staticmethod
    def _to_score(output: Any) -> SentimentScore:
        """Map one pipeline result (a list of {label, score} dicts) to a domain score."""
        scores = output if isinstance(output, list) else [output]
        best: dict[str, Any] | None = None
        for entry in scores:
            if not isinstance(entry, dict):
                continue
            if best is None or float(entry.get("score", 0.0)) > float(best.get("score", 0.0)):
                best = entry
        if best is None:
            msg = "sentiment pipeline returned no usable classification"
            raise ModelUnavailableError(msg)

        label = _LABELS.get(str(best.get("label", "")).strip().lower(), SentimentLabel.NEUTRAL)
        confidence = min(max(float(best.get("score", 0.0)), 0.0), 1.0)
        return SentimentScore(label=label, confidence=round(confidence, 4))
