"""AI evaluation run #13, seed 1 (2026-11-16, v3), rebuilt as offline boards.

`backend/artifacts/live-v3-claude-sonnet55-seeds1-3-2026-11-16/` holds the
record; its `seed1-review-causes.md` traces the owner's seed-1 review to what
each seat's board memo did and did not say. The fixes are pinned at the real
moments of that game, so the board is rebuilt here from the committed
`final_state` instead of being hand-dealt:

* `board(day)`            -- the morning of `day`, before its first public line;
* `board(day, "m75")`     -- right after message m75 (its claims included);
* `board(day, "vote")`    -- after the day's last public line, before ballots;
* `board(day, "night")`   -- after the day's execution, with that day's ballots
  and the night's private chat, before the night resolves.

What a seat could not have known yet is left out: results of nights not yet
resolved, the medium's read of an execution before the night that delivers it,
ballots of the day still being argued. Messages and claims are cut at the same
point, so `board(2, "m32")` is the board ケント saw while writing m33's reply.

The deal: ケント seer; ハルト, カイト, ダイキ wolves; アオイ madman; ソウタ fox;
ミオ hunter; アカリ medium; マコト and ホノカ freemasons; ノゾミ the first
victim. ケント was guarded every night, the wolves bit him on nights 1, 2 and
4 (no corpse) and アカリ on night 3. ユイ, アオイ, ダイキ, ハルト and カイト
were hanged on days 1-5, and the fox won when カイト, the last wolf, died.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

from app.ai.reasoning.board_memo import PRIVATE_HEADING, SELF, render_board_memo
from app.ai.reasoning.observations import ObservationSet
from app.ai.reasoning.perspectives import CommonPublicPerspective, PlayerPrivatePerspective
from app.ai.reasoning.solver import build_solver
from app.engine.phases import Phase
from app.engine.roles import RoleName
from app.engine.speech_events import SpeechEvent, SpeechEventType
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

TRANSCRIPT = (
    Path(__file__).resolve().parents[3]
    / "artifacts"
    / "live-v3-claude-sonnet55-seeds1-3-2026-11-16"
    / "seed1"
    / "transcript-seed1.json"
)

# Seats by name, so a test reads like the game it is about.
MAKOTO, AKARI, HARUTO, YUI, SOTA, MIO, REN, SAKURA, KAITO = (
    "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8",
)
NOZOMI, RIKU, TSUMUGI, DAIKI, HONOKA, SHION, AOI, KENT = (
    "p9", "p10", "p11", "p12", "p13", "p14", "p15", "p16",
)
WOLVES = (HARUTO, KAITO, DAIKI)


@cache
def record() -> dict[str, Any]:
    with TRANSCRIPT.open(encoding="utf-8") as handle:
        data: dict[str, Any] = json.load(handle)
    return data


def board(day: int, at: str = "morning") -> GameState:
    """The board of `day` at `at`: 'morning', 'vote', 'night' or a message id."""
    final = record()["final_state"]
    players = {
        entry["player_id"]: PlayerState(
            player_id=entry["player_id"], name=entry["name"], role=RoleName(entry["role"])
        )
        for entry in final["players"]
    }
    phase = {"vote": Phase.VOTING, "night": Phase.NIGHT}.get(at, Phase.DISCUSSION)
    state = GameState(session_id="run13-seed1", players=players, phase=phase, day=day)

    for entry in final["death_records"]:
        executed_today = entry["day"] == day and entry["cause"] == DeathCause.EXECUTED.value
        if entry["day"] < day or (at == "night" and executed_today):
            cause = DeathCause(entry["cause"])
            player = players[entry["player_id"]]
            player.alive = False
            player.death_cause = cause
            player.death_day = entry["day"]
            state.death_records.append(
                DeathRecord(player_id=entry["player_id"], cause=cause, day=entry["day"])
            )
            if cause is DeathCause.FIRST_VICTIM:
                state.first_victim_id = entry["player_id"]

    # Night records are dated by the night; a night is resolved by the morning
    # after it. The medium reads day N's execution during night N.
    state.divine_records = [DivineRecord(**r) for r in final["divine_records"] if r["day"] < day]
    state.medium_records = [MediumRecord(**r) for r in final["medium_records"] if r["day"] < day]
    state.guard_records = [GuardRecord(**r) for r in final["guard_records"] if r["day"] < day]
    state.attack_records = [AttackRecord(**r) for r in final["attack_records"] if r["day"] < day]
    state.vote_records = [
        VoteRecord(**r)
        for r in final["vote_records"]
        if r["day"] < day or (at == "night" and r["day"] == day)
    ]

    order = {message["message_id"]: index for index, message in enumerate(final["chat_log"])}
    cut = _cut(final["chat_log"], order, day, at)

    def seen(message_id: str, message_day: int) -> bool:
        if message_day != day:
            return message_day < day
        return message_id in order and order[message_id] <= cut

    for message in final["chat_log"]:
        if seen(message["message_id"], message["day"]):
            state.chat_log.append(
                ChatMessage(
                    message_id=message["message_id"],
                    author_id=message["author_id"],
                    content=message["content"],
                    channel=ChatChannel(message["channel"]),
                    day=message["day"],
                    reply_to=message.get("reply_to"),
                    quote=message.get("quote"),
                    references=message.get("references") or [],
                )
            )
    for event in final["speech_events"]:
        if seen(event["source_message_id"], event["day"]):
            state.speech_events.append(
                SpeechEvent(
                    event_id=event["event_id"],
                    source_message_id=event["source_message_id"],
                    actor_id=event["actor_id"],
                    event_type=SpeechEventType(event["event_type"]),
                    day=event["day"],
                    target_id=event.get("target_id"),
                    role=RoleName(event["role"]) if event.get("role") else None,
                    result_is_werewolf=event.get("result_is_werewolf"),
                    referenced_day=event.get("referenced_day"),
                    confidence=event.get("confidence", 1.0),
                )
            )
    return state


def _cut(chat_log: list[dict[str, Any]], order: dict[str, int], day: int, at: str) -> int:
    """Index of the last message of `day` that is on the board at `at`."""
    if at == "morning":
        return -1
    todays = [message for message in chat_log if message["day"] == day]
    if at == "night":
        return max((order[message["message_id"]] for message in todays), default=-1)
    if at == "vote":
        public = [m for m in todays if m["channel"] == ChatChannel.PUBLIC.value]
        return order[public[-1]["message_id"]] if public else -1
    if at not in order:
        raise KeyError(f"no message {at} in the record")
    return order[at]


def memo(state: GameState, seat: str) -> str:
    """The memo the seat reads, through the same solvers the runtime builds."""
    observations = ObservationSet.from_state(state)
    return render_board_memo(
        state,
        seat,
        observations=observations,
        public_solver=build_solver(observations, CommonPublicPerspective()),
        seat_solver=build_solver(observations, PlayerPrivatePerspective(seat)),
    )


def public_part(text: str) -> str:
    """Everything above the private section, minus the header naming the seat."""
    return text.split(PRIVATE_HEADING)[0].partition("\n")[2]


def private_part(text: str) -> str:
    return text.split(PRIVATE_HEADING)[1]


def shared_part(text: str, seat: str) -> str:
    """The public part with the seat's 自分 written back as its name: what
    every seat's memo has in common (the memo names the reader 自分, 2-C1)."""
    assert text.partition("\n")[0].endswith(f"あなた: {name(seat)}")
    return public_part(text).replace(SELF, name(seat))


def name(seat: str) -> str:
    return str(record()["names"][seat])
