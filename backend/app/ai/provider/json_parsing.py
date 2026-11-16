"""The permissive JSON extraction both real providers fall back on.

A model that was not (or could not be) held to a JSON-schema grammar still
usually answers with the right object -- wrapped in a ```json fence, or with a
sentence before or after it. Throwing that turn away costs a whole retry, so
the text is tried in three increasingly forgiving ways, each one still
validated against the Pydantic schema:

  1. the raw text as-is,
  2. the first fenced ```json block,
  3. the slice from the first `{` to the last `}`.

Only the *location* of the object is guessed, never its content: a field the
schema rejects is still rejected. Shared rather than copied so the luna and
anthropic providers cannot drift into salvaging different things.
"""

from __future__ import annotations

import re

from pydantic import ValidationError

from app.ai.provider.base import SchemaT

_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def parse_strict(content: str | None, response_schema: type[SchemaT]) -> SchemaT | None:
    """Validate the text exactly as given, or None."""
    if not content:
        return None
    try:
        return response_schema.model_validate_json(content)
    except (ValidationError, ValueError):
        return None


def parse_permissive(content: str | None, response_schema: type[SchemaT]) -> SchemaT | None:
    """Find and validate the JSON object in the text (see module docstring), or None."""
    if not content:
        return None

    direct = parse_strict(content, response_schema)
    if direct is not None:
        return direct

    fenced = _FENCED_JSON_RE.search(content)
    if fenced is not None:
        candidate = parse_strict(fenced.group(1), response_schema)
        if candidate is not None:
            return candidate

    start, end = content.find("{"), content.rfind("}")
    if start != -1 and end != -1 and end > start:
        return parse_strict(content[start : end + 1], response_schema)

    return None
