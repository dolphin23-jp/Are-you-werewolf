"""Game event notifications, decoupled from GameController's rule logic."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class GameEventType(StrEnum):
    CHAT_MESSAGE = "chat_message"
    PHASE_CHANGED = "phase_changed"
    PLAYER_DIED = "player_died"
    DIVINE_RESULT = "divine_result"
    MEDIUM_RESULT = "medium_result"
    VOTE_RESULT = "vote_result"
    CO_DECLARED = "co_declared"
    VICTORY = "victory"


@dataclass
class GameEvent:
    type: GameEventType
    payload: dict[str, Any] = field(default_factory=dict)
    recipient_id: str | None = None  # None means broadcast to all


Listener = Callable[[GameEvent], None]


class EventBus:
    def __init__(self) -> None:
        self._listeners: list[Listener] = []

    def subscribe(self, listener: Listener) -> None:
        self._listeners.append(listener)

    def unsubscribe(self, listener: Listener) -> None:
        if listener in self._listeners:
            self._listeners.remove(listener)

    def publish(self, event: GameEvent) -> None:
        for listener in list(self._listeners):
            listener(event)
