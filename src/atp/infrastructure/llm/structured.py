"""Coaxing schema-valid JSON out of models that lack native structured output.

The Anthropic adapter gets this for free — the SDK validates against a Pydantic
model server-side. Free and local models mostly do not: support for
``response_format: json_schema`` is patchy, and a model that ignores it will
happily wrap its answer in prose or a markdown fence.

So the schema is enforced three ways, weakest last:

1. the provider's own structured-output parameter, when it has one;
2. the schema restated in the prompt, which every model can at least read;
3. this module, which extracts and validates whatever came back.

Step 3 is the one that actually guarantees the contract. Anything that fails
validation raises ``LLMGenerationError``, so a sloppy model degrades into a
recorded agent failure rather than a malformed report reaching a trader.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ValidationError

from atp.domain.errors import LLMGenerationError

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def schema_instructions[T: BaseModel](response_model: type[T]) -> str:
    """A prompt fragment restating the required shape, for models that ignore params."""
    schema = json.dumps(response_model.model_json_schema(), indent=2, sort_keys=True)
    return (
        "Respond with a single JSON object and nothing else - no prose, no "
        "explanation outside the JSON, no markdown code fence. It must validate "
        f"against this JSON Schema:\n\n{schema}"
    )


def extract_json(text: str) -> str:
    """Pull the JSON object out of a response that may be fenced or padded with prose."""
    stripped = text.strip()
    if fenced := _FENCE_RE.search(stripped):
        return fenced.group(1).strip()

    # Fall back to the outermost braces: models like to prepend "Here is the JSON:".
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end > start:
        return stripped[start : end + 1]
    return stripped


def parse_structured[T: BaseModel](response_model: type[T], text: str, *, model: str) -> T:
    """Validate a raw completion into ``response_model`` or raise."""
    if not text.strip():
        msg = f"model '{model}' returned an empty response"
        raise LLMGenerationError(msg)

    payload = extract_json(text)
    try:
        data: Any = json.loads(payload)
    except json.JSONDecodeError as exc:
        preview = payload[:200].replace("\n", " ")
        msg = f"model '{model}' did not return JSON ({exc}); got: {preview}"
        raise LLMGenerationError(msg) from exc

    try:
        return response_model.model_validate(data)
    except ValidationError as exc:
        msg = (
            f"model '{model}' returned JSON that does not match "
            f"{response_model.__name__}: {exc.error_count()} validation error(s). "
            "Smaller models often omit required fields such as evidence or "
            "invalidation_conditions."
        )
        raise LLMGenerationError(msg) from exc
