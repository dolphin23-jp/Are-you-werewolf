"""Event-driven logical-time scheduler for public discussion.

A caller asks every eligible player for a fresh :class:`SpeakIntent`, executes
only the earliest one, appends the resulting event, and then asks everybody
again. Nothing is pre-committed across another player's speech, which lets a
wolf abandon a planned claim after seeing a new CO, or a medium wait for more
information before speaking.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Mapping

from app.training.actions import SpeechBundle, TimingBucket


@dataclass(frozen=True)
class SpeakIntent:
    timing: TimingBucket
    bundle: SpeechBundle | None = None

    @property
    def will_speak(self) -> bool:
        return self.timing is not TimingBucket.HOLD and self.bundle is not None


@dataclass(frozen=True)
class ScheduledSpeaker:
    player_id: str
    timing: TimingBucket
    discussion_tick: int


class EventDrivenDiscussionScheduler:
    """Select one speaker, then force replanning before selecting another."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self.discussion_tick = 0

    def reset(self) -> None:
        self.discussion_tick = 0

    def select_next(self, intents: Mapping[str, SpeakIntent]) -> ScheduledSpeaker | None:
        eligible = [(player_id, intent) for player_id, intent in intents.items() if intent.will_speak]
        if not eligible:
            return None

        earliest = min(intent.timing for _, intent in eligible)
        tied = [player_id for player_id, intent in eligible if intent.timing == earliest]
        player_id = self._rng.choice(sorted(tied))
        return ScheduledSpeaker(
            player_id=player_id,
            timing=earliest,
            discussion_tick=self.discussion_tick,
        )

    def record_emitted_event(self) -> int:
        """Advance logical time after exactly one speech turn is committed."""
        self.discussion_tick += 1
        return self.discussion_tick
