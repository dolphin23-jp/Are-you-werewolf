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
    TYPING_CHANGED = "typing_changed"
    VICTORY = "victory"


@dataclass
class GameEvent:
    """One notification, and who may receive it.

    Delivery is decided here, by the engine that knows the roles, and not by
    the transport: a WebSocket hub only sees player ids, so an event that does
    not say who it is for goes to every seat. That is right for public events
    and a leak for anything else -- a wolf-channel message pushed to every
    socket tells a villager with devtools open who the wolves are.

    `recipient_id` addresses one seat (a seer's divination). `recipient_ids`
    addresses a group (a private channel's members). With neither set the
    event is public. An empty `recipient_ids` means *nobody*, never everyone:
    a private channel whose members are all gone must not fall back to a
    broadcast.
    """

    type: GameEventType
    payload: dict[str, Any] = field(default_factory=dict)
    recipient_id: str | None = None
    recipient_ids: tuple[str, ...] | None = None

    @property
    def is_public(self) -> bool:
        return self.recipient_id is None and self.recipient_ids is None

    def audience(self) -> frozenset[str] | None:
        """The seats allowed to receive this event, or None when it is public."""
        if self.is_public:
            return None
        ids = set(self.recipient_ids or ())
        if self.recipient_id is not None:
            ids.add(self.recipient_id)
        return frozenset(ids)


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
