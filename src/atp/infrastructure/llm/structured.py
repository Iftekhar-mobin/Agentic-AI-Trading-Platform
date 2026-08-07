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
validation raises ``SchemaViolationError``, so a sloppy model degrades into a
recorded agent failure rather than a malformed report reaching a trader.

A failure at step 3 is not always terminal, though. Models that drop one
required field usually supply it when told exactly which field and why, so the
error carries enough detail for a caller to ask for a correction — see
``repair_prompt``. Adapters decide whether to spend that second call.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ValidationError

from atp.domain.errors import LLMGenerationError

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class SchemaViolationError(LLMGenerationError):
    """A response that did not satisfy the requested schema.

    Subclasses ``LLMGenerationError`` so existing handling is unchanged: the
    orchestrator still records a failed agent. ``detail`` is the extra material
    an adapter needs to ask the model to correct itself.
    """

    def __init__(self, message: str, *, response: str, detail: str) -> None:
        super().__init__(message)
        self.response = response
        """The raw completion, replayed to the model as its own prior turn."""
        self.detail = detail
        """What was wrong, phrased for the model rather than for a log reader."""


def repair_prompt[T: BaseModel](response_model: type[T], detail: str) -> str:
    """A corrective follow-up naming exactly what was wrong.

    Worth one extra call: the common free-model failure is a single omitted
    field, and quoting the specific error fixes it far more often than simply
    restating the schema — which the model has already seen and ignored once.
    """
    return (
        f"That response did not validate against {response_model.__name__}:\n\n"
        f"{detail}\n\n"
        "Return the corrected JSON object, complete and on its own. Every "
        "required field must be present - do not omit evidence or "
        "invalidation_conditions. No prose, no markdown fence."
    )


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
        detail = f"The output was not valid JSON: {exc}"
        raise SchemaViolationError(msg, response=text, detail=detail) from exc

    try:
        return response_model.model_validate(data)
    except ValidationError as exc:
        msg = (
            f"model '{model}' returned JSON that does not match "
            f"{response_model.__name__}: {exc.error_count()} validation error(s). "
            "Smaller models often omit required fields such as evidence or "
            "invalidation_conditions."
        )
        raise SchemaViolationError(msg, response=text, detail=_violations(exc)) from exc


def _violations(exc: ValidationError) -> str:
    """Pydantic's errors as a short bulleted list the model can act on.

    Truncated: a model that produced twenty errors will not be rescued by
    reading all twenty, and the prompt still has an agent's context in it.
    """
    lines = [
        f"- {'.'.join(str(part) for part in error['loc']) or '(root)'}: {error['msg']}"
        for error in exc.errors()[:10]
    ]
    remaining = exc.error_count() - len(lines)
    if remaining > 0:
        lines.append(f"- ...and {remaining} more")
    return "\n".join(lines)
