"""The first live Claude game (2026-11-16, v3, seed 1), rebuilt at a moment in it.

The read of that game (`artifacts/live-v3-claude-opus55-seed1-2026-11-16/
analysis.md`) found its defects in the record itself: a vote change filed as
unexplained although the seat cited an argument made after its turn, an ally
vote nobody audited, a blacked villager who never got the floor. A hand-built
board only approximates those positions; the record holds them exactly. These
helpers put the committed transcript back into a `GameState` as it stood when
a given day's ballots were cast, so a test can hand the coordinator the very
board it misjudged.

Nothing here reaches a model or the network: the transcript is a file in the
repository.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.engine.phases import Phase
from app.engine.roles import RoleName
from app.engine.speech_events import SpeechEventType
from app.engine.state import (
    AttackRecord,
    ChatChannel,
    ChatMessage,
    DeathCause,
    DeathRecord,
    DivineRecord,
    GameState,
    GuardRecord,
    MediumRecord,
    PlayerState,
    VoteRecord,
)

ARTIFACT_DIR = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "live-v3-claude-opus55-seed1-2026-11-16"
)
TRANSCRIPT_PATH = ARTIFACT_DIR / "transcript-seed1.json"

# Every seat in that game was an AI; seat p0 (マコト) was the freemason leader.
AI_IDS = [f"p{i}" for i in range(17)]


@lru_cache(maxsize=1)
def load_transcript() -> dict[str, Any]:
    with TRANSCRIPT_PATH.open(encoding="utf-8") as stream:
        data: dict[str, Any] = json.load(stream)
    return data


def name_to_id() -> dict[str, str]:
    return {name: pid for pid, name in load_transcript()["names"].items()}


def state_at_vote(day: int) -> GameState:
    """The board when day `day`'s first ballot was cast.

    Everything public from that day's discussion is in (it all happens before
    the vote); that night's private chat, deaths and night actions are not.
    Night `n` events carry day `n`, so the previous nights are `< day`.
    """
    return replay_at_vote(load_transcript(), day, session_id="first-claude-game")


def replay_at_vote(data: dict[str, Any], day: int, *, session_id: str) -> GameState:
    """`state_at_vote` for any committed transcript, given as its parsed JSON."""
    final = data["final_state"]
    players = {
        raw["player_id"]: PlayerState(
            player_id=raw["player_id"],
            name=raw["name"],
            role=RoleName(raw["role"]),
            is_human=bool(raw.get("is_human", False)),
        )
        for raw in final["players"]
    }
    state = GameState(session_id=session_id, players=players, phase=Phase.VOTING)
    for record in final["death_records"]:
        if record["day"] >= day:
            continue
        cause = DeathCause(record["cause"])
        state.death_records.append(DeathRecord(record["player_id"], cause, record["day"]))
        player = state.players[record["player_id"]]
        player.alive = False
        player.death_cause = cause
        player.death_day = record["day"]
        if cause is DeathCause.FIRST_VICTIM:
            state.first_victim_id = record["player_id"]
    numbers = [0]
    for raw in final["chat_log"]:
        channel = ChatChannel(raw["channel"])
        public = channel is ChatChannel.PUBLIC
        if raw["day"] > day or (not public and raw["day"] >= day):
            continue
        state.chat_log.append(
            ChatMessage(
                message_id=raw["message_id"],
                author_id=raw["author_id"],
                content=raw["content"],
                channel=channel,
                day=raw["day"],
                reply_to=raw.get("reply_to"),
                quote=raw.get("quote"),
                references=list(raw.get("references") or []),
            )
        )
        numbers.append(int(raw["message_id"].lstrip("m")))
    state.next_message_number = max(numbers) + 1
    for raw in final["speech_events"]:
        if raw["day"] > day:
            continue
        # The event log stamps the day from the state, so it is set per event.
        state.day = raw["day"]
        state.append_speech_event(
            raw["actor_id"],
            SpeechEventType(raw["event_type"]),
            source_message_id=raw["source_message_id"],
            target_id=raw["target_id"],
            role=RoleName(raw["role"]) if raw["role"] else None,
            result_is_werewolf=raw["result_is_werewolf"],
            referenced_day=raw["referenced_day"],
        )
    state.divine_records = [
        DivineRecord(r["seer_id"], r["target_id"], r["day"], r["is_werewolf"])
        for r in final["divine_records"]
        if r["day"] < day
    ]
    state.medium_records = [
        MediumRecord(r["medium_id"], r["target_id"], r["day"], r["is_werewolf"])
        for r in final["medium_records"]
        if r["day"] < day
    ]
    state.guard_records = [
        GuardRecord(r["hunter_id"], r["target_id"], r["day"])
        for r in final["guard_records"]
        if r["day"] < day
    ]
    state.attack_records = [
        AttackRecord(r["wolf_id"], r["target_id"], r["day"], r["succeeded"])
        for r in final["attack_records"]
        if r["day"] < day
    ]
    state.vote_records = [
        VoteRecord(r["voter_id"], r["target_id"], r["day"], r["round"])
        for r in final["vote_records"]
        if r["day"] < day
    ]
    state.day = day
    state.phase = Phase.VOTING
    return state


def ballots(day: int) -> dict[str, str]:
    """Who each seat voted for on `day` (the game had no runoffs)."""
    return {
        r["voter_id"]: r["target_id"]
        for r in load_transcript()["final_state"]["vote_records"]
        if r["day"] == day
    }


class BallotBox:
    """The one controller call `_cast_vote` makes, against a replayed state.

    The engine's vote manager would also check the phase and the target; the
    replayed ballots were legal when they were cast, so this only records them.
    """

    def __init__(self, state: GameState) -> None:
        self.state = state

    def vote(self, voter_id: str, target_id: str) -> None:
        self.state.pending_votes[voter_id] = target_id
