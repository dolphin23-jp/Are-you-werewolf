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

        recipients = [event.recipient_id] if event.recipient_id else list(self._connections.keys())
        payload = {"type": event.type.value, "payload": to_jsonable(event.payload)}
        for pid in recipients:
            for ws in self._connections.get(pid, []):
                loop.create_task(_safe_send(ws, payload))


async def _safe_send(ws: WebSocket, payload: dict[str, Any]) -> None:
    try:
        await ws.send_json(payload)
    except Exception:
        pass
