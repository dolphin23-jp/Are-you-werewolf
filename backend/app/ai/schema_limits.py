"""How complex a structured-output schema is, counted the way Anthropic limits it.

Anthropic compiles every `output_config.format` JSON schema into a grammar and
refuses a request whose schema is too large to compile: at most 24 optional
parameters and at most 16 parameters with union types per request, past which
the API answers 400 "Schema is too complex for compilation"
(https://platform.claude.com/docs/en/build-with-claude/structured-outputs,
limits checked 2026-11-15). A provider cannot recover from that by retrying,
and the failure only shows up against the live API -- so the counts are kept
here, as pure code the offline tests run against every schema we send.

The counter reads an already transformed schema (`anthropic.transform_schema`
output, or any JSON schema of the same shape) and applies these rules:

- `$ref` is resolved into `$defs` and counted where it is used, so a definition
  referenced from two properties counts twice -- the conservative reading;
- every property not in its object's `required` list is optional, and nested
  objects' properties count as well;
- an array's item schema counts once, however many items the answer may hold;
- a property whose schema has `anyOf` or a list-valued `type` is a union, and
  so is an array item schema of that shape (again the conservative reading).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

MAX_OPTIONAL_PARAMETERS = 24
MAX_UNION_PARAMETERS = 16

_REF_PREFIX = "#/$defs/"


def count_schema_parameters(schema: dict[str, Any]) -> tuple[int, int]:
    """Return `(optional, union)` parameter counts for one request schema."""
    defs = schema.get("$defs") or {}
    counts = [0, 0]
    _walk(schema, defs, counts, expanding=())
    return counts[0], counts[1]


def check_within_limits(model_cls: type[BaseModel]) -> bool:
    """Whether `model_cls`, as the Anthropic SDK would send it, compiles.

    The SDK is imported here rather than at module load, so the pure counter
    stays usable where only the schema, not the SDK, is at hand.
    """
    from anthropic import transform_schema

    optional, union = count_schema_parameters(transform_schema(model_cls))
    return optional <= MAX_OPTIONAL_PARAMETERS and union <= MAX_UNION_PARAMETERS


def _walk(
    node: dict[str, Any],
    defs: dict[str, Any],
    counts: list[int],
    *,
    expanding: tuple[str, ...],
) -> None:
    """Count the parameters reachable from `node` (not `node` itself).

    `expanding` holds the definitions being expanded on the current path, so a
    self-referencing schema is counted once instead of recursing forever.
    """
    ref = node.get("$ref")
    if isinstance(ref, str):
        if not ref.startswith(_REF_PREFIX):
            raise ValueError(f"unsupported $ref {ref!r}: only local $defs are resolved")
        name = ref[len(_REF_PREFIX) :]
        if name in expanding:
            return
        if name not in defs:
            raise ValueError(f"$ref {ref!r} names no definition in $defs")
        _walk(defs[name], defs, counts, expanding=(*expanding, name))
        return

    for key in ("anyOf", "allOf"):
        for variant in node.get(key) or ():
            _walk(variant, defs, counts, expanding=expanding)

    properties = node.get("properties")
    if isinstance(properties, dict):
        required = set(node.get("required") or ())
        for name, prop in properties.items():
            if name not in required:
                counts[0] += 1
            if _is_union(prop):
                counts[1] += 1
            _walk(prop, defs, counts, expanding=expanding)

    items = node.get("items")
    if isinstance(items, dict):
        if _is_union(items):
            counts[1] += 1
        _walk(items, defs, counts, expanding=expanding)


def _is_union(node: dict[str, Any]) -> bool:
    return bool(node.get("anyOf")) or isinstance(node.get("type"), list)
