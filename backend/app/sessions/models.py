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

    def touch(self) -> None:
        self.last_active_at = time.time()
