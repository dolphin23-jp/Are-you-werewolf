"""Who is making a model call, and which call it is, without threading it through.

A provider sees a system prompt and messages, never the seat that sent them,
and a call's `CallRecord` is written deep inside the provider. To join a call's
record (latency, tokens, cost, stop reason) to the seat that made it and to the
prompt it sent, the caller sets these values for the duration of the call and
`MetricsCollector.record` stamps them onto the record. A `ContextVar` follows
the call through every `await` (and into any task it starts), and concurrent
calls -- a vote round runs several at once -- each see their own values.

Nothing reads these to make a decision; they only label records.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class CallContext:
    seat_id: str | None = None
    call_index: int | None = None


_EMPTY = CallContext()
_CURRENT: ContextVar[CallContext | None] = ContextVar("werewolf_call_context", default=None)


def current_call_context() -> CallContext:
    return _CURRENT.get() or _EMPTY


@contextmanager
def calling_as(*, seat_id: str | None = None, call_index: int | None = None) -> Iterator[None]:
    """Label the calls made inside this block. A value left as None keeps the
    one already set by an outer block."""
    outer = current_call_context()
    changes: dict[str, object] = {}
    if seat_id is not None:
        changes["seat_id"] = seat_id
    if call_index is not None:
        changes["call_index"] = call_index
    token = _CURRENT.set(replace(outer, **changes) if changes else outer)  # type: ignore[arg-type]
    try:
        yield
    finally:
        _CURRENT.reset(token)
