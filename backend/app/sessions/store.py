"""SessionStore: fixes the prior implementation's single-global-session gap
by keying game sessions on an explicit session_id from the start, so
multiple concurrent games are supported in-process."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.sessions.models import GameSession


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

    def __init__(self) -> None:
        self._sessions: dict[str, GameSession] = {}

    def create(self, session: GameSession) -> None:
        self._sessions[session.session_id] = session

    def get(self, session_id: str) -> GameSession | None:
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def list_ids(self) -> list[str]:
        return list(self._sessions.keys())


_store: InMemorySessionStore | None = None


def get_session_store() -> SessionStore:
    global _store
    if _store is None:
        _store = InMemorySessionStore()
    return _store
