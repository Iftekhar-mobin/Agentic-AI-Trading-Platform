"""Applies sampled parameter values to a strategy definition.

Pure function over the declarative schema: dump the model, set values at the
bound dotted paths, re-validate. Re-validation means a sampled combination can
never produce a structurally invalid strategy — it raises instead.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, MutableSequence
from typing import TYPE_CHECKING, Any

from atp.domain.models.strategy import StrategyDefinition

if TYPE_CHECKING:
    from atp.domain.models.optimization import SearchParameter


def apply_parameters(
    strategy: StrategyDefinition,
    parameters: tuple[SearchParameter, ...],
    values: Mapping[str, float],
) -> StrategyDefinition:
    from atp.domain.models.optimization import ParameterKind

    # JSON mode turns tuples into mutable lists so paths can be set in place.
    dump = strategy.model_dump(mode="json")
    for parameter in parameters:
        raw = values[parameter.name]
        value: float | int = int(raw) if parameter.kind is ParameterKind.INT else float(raw)
        for path in parameter.paths:
            _set_path(dump, path, value)
    return StrategyDefinition.model_validate(dump)


def _set_path(root: MutableMapping[str, Any], path: str, value: float | int) -> None:
    segments = path.split(".")
    node: Any = root
    for segment in segments[:-1]:
        node = _step(node, segment, path)
    leaf = segments[-1]
    if isinstance(node, MutableMapping):
        if leaf not in node:
            msg = f"path '{path}': no field '{leaf}'"
            raise KeyError(msg)
        node[leaf] = value
    elif isinstance(node, MutableSequence):
        node[int(leaf)] = value
    else:
        msg = f"path '{path}': cannot set '{leaf}' on {type(node).__name__}"
        raise TypeError(msg)


def _step(node: Any, segment: str, path: str) -> Any:
    if isinstance(node, MutableMapping):
        if segment not in node:
            msg = f"path '{path}': no field '{segment}'"
            raise KeyError(msg)
        return node[segment]
    if isinstance(node, MutableSequence):
        index = int(segment)
        if index >= len(node):
            msg = f"path '{path}': index {index} out of range"
            raise IndexError(msg)
        return node[index]
    msg = f"path '{path}': cannot descend into {type(node).__name__} at '{segment}'"
    raise TypeError(msg)
