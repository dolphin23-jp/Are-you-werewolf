"""Bridges the synchronous engine EventBus to async WebSocket pushes."""

from __future__ import annotations

import asyncio
from enum import Enum
from typing import Any

from fastapi import WebSocket

from app.engine.events import GameEvent


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


class SessionWSHub:
    """One hub per game session; player_id -> connected sockets."""

    def __init__(self) -> None:
        self._connections: dict[str, list[WebSocket]] = {}
        # The loop keeps only a weak reference to a task; one nobody holds can
        # be collected before its frame is sent.
        self._sends: set[asyncio.Task[None]] = set()

    async def connect(self, player_id: str, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.setdefault(player_id, []).append(ws)

    def disconnect(self, player_id: str, ws: WebSocket) -> None:
        conns = self._connections.get(player_id)
        if conns and ws in conns:
            conns.remove(ws)

    def on_event(self, event: GameEvent) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # GameController is called from a few sync route handlers too
            # (FastAPI dispatches plain `def` routes on a worker thread with
            # no event loop). Silently drop the push in that case -- the
            # WS client's polling fallback (useGamePolling) still catches
            # up, and this must never crash the request that triggered it.
            return

        # The engine decides who an event is for (`GameEvent.audience`); the hub
        # only honours it. A public event goes to every connected seat; any
        # other event only to the seats named, so a private channel's frames
        # never reach a socket outside it.
        audience = event.audience()
        recipients = list(self._connections) if audience is None else sorted(audience)
        payload = {"type": event.type.value, "payload": to_jsonable(event.payload)}
        for pid in recipients:
            for ws in self._connections.get(pid, []):
                task = loop.create_task(_safe_send(ws, payload))
                self._sends.add(task)
                task.add_done_callback(self._sends.discard)


async def _safe_send(ws: WebSocket, payload: dict[str, Any]) -> None:
    try:
        await ws.send_json(payload)
    except Exception:
        pass
