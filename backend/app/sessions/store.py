"""SessionStore: fixes the prior implementation's single-global-session gap
by keying game sessions on an explicit session_id from the start, so
multiple concurrent games are supported in-process."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

from app.sessions.models import GameSession

# Nothing ever removed a game, and creating one needs no password by default:
# each started game holds tens of MB, so the process only grew. A personal
# server rarely has more than a couple of live games.
MAX_SESSIONS = 20
IDLE_TTL_SECONDS = 6 * 60 * 60


class SessionStore(ABC):
    @abstractmethod
    def create(self, session: GameSession) -> None: ...

    @abstractmethod
    def get(self, session_id: str) -> GameSession | None: ...

    @abstractmethod
    def delete(self, session_id: str) -> None: ...

    @abstractmethod
    def list_ids(self) -> list[str]: ...


class InMemorySessionStore(SessionStore):
    """v1 persistence decision: in-memory only, explicit and documented
    (not an accidental limitation). A future SQLite-backed store can
    implement the same interface without touching callers."""

    def __init__(
        self, max_sessions: int = MAX_SESSIONS, idle_ttl_seconds: float = IDLE_TTL_SECONDS
    ) -> None:
        self._sessions: dict[str, GameSession] = {}
        self._max_sessions = max_sessions
        self._idle_ttl_seconds = idle_ttl_seconds

    def create(self, session: GameSession) -> None:
        self._evict(time.time())
        self._sessions[session.session_id] = session

    def _evict(self, now: float) -> None:
        """Drop idle games, then the least recently used, to make room."""
        for session_id, session in list(self._sessions.items()):
            if now - session.last_active_at > self._idle_ttl_seconds:
                self.delete(session_id)
        while self._sessions and len(self._sessions) >= self._max_sessions:
            oldest = min(self._sessions.values(), key=lambda item: item.last_active_at)
            self.delete(oldest.session_id)

    def get(self, session_id: str) -> GameSession | None:
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session is None:
            return
        for task in (session.discussion_advance_task, *session.private_reply_tasks.values()):
            if task is not None and not task.done():
                task.cancel()

    def list_ids(self) -> list[str]:
        return list(self._sessions.keys())


_store: InMemorySessionStore | None = None


def get_session_store() -> SessionStore:
    global _store
    if _store is None:
        _store = InMemorySessionStore()
    return _store
