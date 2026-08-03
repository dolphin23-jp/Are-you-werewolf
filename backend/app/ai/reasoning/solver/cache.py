"""Memoization for solver queries.

The same question gets asked many times in a turn -- every AI wants "can X be a
wolf" about the same board -- and the answer only changes when the board, the
viewpoint, the assumptions or the kind of question changes. All four are in the
key, so a cache hit is genuinely the same query and not a near-miss.

Bounded on purpose: an unbounded dict on a long game is a slow leak, and the
oldest entries are the ones whose board is furthest behind.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

DEFAULT_CAPACITY = 2048


@dataclass(frozen=True)
class QueryKey:
    board_version: str
    perspective_id: str
    constraint_signature: str
    query_kind: str
    assumptions: tuple[str, ...]


class SolverCache:
    def __init__(self, capacity: int = DEFAULT_CAPACITY) -> None:
        self._capacity = max(1, capacity)
        self._entries: OrderedDict[QueryKey, Any] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.query_seconds = 0.0

    def get(self, key: QueryKey) -> Any | None:
        if key not in self._entries:
            self.misses += 1
            return None
        self.hits += 1
        self._entries.move_to_end(key)
        return self._entries[key]

    def put(self, key: QueryKey, value: Any) -> None:
        self._entries[key] = value
        self._entries.move_to_end(key)
        while len(self._entries) > self._capacity:
            self._entries.popitem(last=False)

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()
        self.hits = 0
        self.misses = 0
        self.query_seconds = 0.0
