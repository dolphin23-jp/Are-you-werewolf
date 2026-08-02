"""Per-game session wrapper: ties a GameController to a human player and
(once M3 wires it up) an AI coordinator driving the other 16 seats."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from app.api.ws_hub import SessionWSHub
from app.engine.game import GameController
from app.eval.transcript import TranscriptRecorder


@dataclass
class DiscussionRoundState:
    day: int
    order: list[str]
    cursor: int = 0
    stage: str = "immediate"
    speech_counts: dict[str, int] = field(default_factory=dict)
    reply_queue: list[str] = field(default_factory=list)
    queued: set[str] = field(default_factory=set)
    outputs: list[tuple[str, Any]] = field(default_factory=list)
    major_targets: list[str] = field(default_factory=list)
    awaiting_human: bool = False
    awaiting_since: float | None = None
    immediate_count: int = 0
    max_total: int = 0
    complete: bool = False
    summary_done: bool = False
    major_targets_ready: bool = False


@dataclass
class GameSession:
    session_id: str
    controller: GameController
    human_id: str
    ai_player_ids: list[str]
    ws_hub: SessionWSHub
    coordinator: Any | None = None
    transcript_recorder: TranscriptRecorder | None = None
    created_at: float = field(default_factory=time.time)
    last_active_at: float = field(default_factory=time.time)
    # Serializes discussion rounds so rapid-fire human chat messages queue
    # sequential AI rounds instead of interleaving overlapping ones.
    discussion_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    discussion_round: DiscussionRoundState | None = None
    discussion_advance_task: asyncio.Task[Any] | None = None

    def touch(self) -> None:
        self.last_active_at = time.time()
