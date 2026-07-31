"""Optional HTTP Basic gate for a personal internet-facing deployment."""

from __future__ import annotations

import base64
import secrets
from collections.abc import Awaitable, Callable
from typing import Any

Scope = dict[str, Any]
Message = dict[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]


class PersonalAccessMiddleware:
    """Protect HTTP and WebSocket traffic when an access password is configured."""

    def __init__(self, app: ASGIApp, password: str, username: str = "werewolf") -> None:
        self.app = app
        self._expected = "Basic " + base64.b64encode(f"{username}:{password}".encode()).decode()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in {"http", "websocket"} or scope.get("path") == "/api/health":
            await self.app(scope, receive, send)
            return

        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        supplied = headers.get(b"authorization", b"").decode("latin-1")
        if secrets.compare_digest(supplied, self._expected):
            await self.app(scope, receive, send)
            return

        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 4401})
            return

        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"text/plain; charset=utf-8"),
                    (b"www-authenticate", b'Basic realm="Are you werewolf", charset="UTF-8"'),
                ],
            }
        )
        await send({"type": "http.response.body", "body": "認証が必要です".encode()})
