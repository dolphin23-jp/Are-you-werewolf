"""The in-memory store drops idle and least-recently-used games."""

from __future__ import annotations

from types import SimpleNamespace

from app.sessions.store import InMemorySessionStore


def _session(session_id: str, last_active_at: float) -> SimpleNamespace:
    return SimpleNamespace(
        session_id=session_id,
        last_active_at=last_active_at,
        discussion_advance_task=None,
        private_reply_tasks={},
    )


def test_the_least_recently_used_game_makes_room():
    store = InMemorySessionStore(max_sessions=2, idle_ttl_seconds=float("inf"))
    store.create(_session("old", 1.0))  # type: ignore[arg-type]
    store.create(_session("new", 2.0))  # type: ignore[arg-type]
    store.create(_session("newest", 3.0))  # type: ignore[arg-type]

    assert sorted(store.list_ids()) == ["new", "newest"]


def test_idle_games_expire():
    store = InMemorySessionStore(max_sessions=10, idle_ttl_seconds=60)
    store.create(_session("abandoned", 0.0))  # type: ignore[arg-type]
    store.create(_session("fresh", 1e12))  # type: ignore[arg-type]

    assert store.list_ids() == ["fresh"]
