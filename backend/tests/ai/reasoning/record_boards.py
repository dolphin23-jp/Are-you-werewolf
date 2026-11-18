"""Offline boards rebuilt from a recorded game's `final_state`.

A run's transcript keeps the whole game; `board_from_record(final, id, day, at)`
cuts it back to one moment, so a fix can be pinned at the real point of the
game it is about:

* `at="morning"`  -- the morning of `day`, before its first public line;
* `at="m75"`      -- right after message m75 (its claims included);
* `at="vote"`     -- after the day's last public line, before ballots;
* `at="night"`    -- after the day's execution, with that day's ballots and the
  night's private chat, before the night resolves.

What a seat could not have known yet is left out: results of nights not yet
resolved, the medium's read of an execution before the night that delivers it,
ballots of the day still being argued. Messages and claims are cut at the same
point.
"""

from __future__ import annotations

from typing import Any

from app.engine.phases import Phase
from app.engine.roles import RoleName, Team
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


def board_from_record(
    final: dict[str, Any], session_id: str, day: int, at: str = "morning"
) -> GameState:
    """The board of `day` at `at`: 'morning', 'vote', 'night' or a message id."""
    players = {
        entry["player_id"]: PlayerState(
            player_id=entry["player_id"], name=entry["name"], role=RoleName(entry["role"])
        )
        for entry in final["players"]
    }
    phase = {"vote": Phase.VOTING, "night": Phase.NIGHT}.get(at, Phase.DISCUSSION)
    state = GameState(session_id=session_id, players=players, phase=phase, day=day)

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
    # The execution that ended the game: the board after it is the final one.
    if final.get("winner") and len(state.death_records) == len(final["death_records"]):
        state.winner = Team(final["winner"])
        state.phase = Phase.GAME_OVER

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
