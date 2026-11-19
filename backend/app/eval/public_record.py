"""The public claim record, against what each seat actually said.

v3's premise is that code guarantees the facts. The public claim record --
every CO, ability result and partner naming in the final state's
`speech_events` -- is written from the lines the seats speak, and every seat
then reads it as fact: the board memo's CO list, the day summary, the human
player's CO panel. A record its speaker never declared is therefore a factual
error made by code, and the table believes it over the words.

Two committed games show what that costs. Run 15, day 5: アオイ said 「リク黒が
外れたのはダイキさんの問題」 -- ダイキ's black on リク -- and the record gained
アオイ→リク=黒. Run 16, day 1: カイト, a wolf told to lurk, said 「共有CO二人
は了解」 -- he acknowledged the two freemason claims -- and the record gained a
third freemason CO for him; the board memo listed 「共有=マコト、ホノカ、カイト」
in 194 of the game's 202 prompts, カイト took up the claim the record gave him,
and a villager was hanged on day 6 for trusting the real freemason.

A record is declared when its own line declares it:

* the line's structured output carries it (`public_claim_role`,
  `public_results` of the utterance the line was posted from), or
* the prose states it as the speaker's own, read by the readers the game
  registers prose with (`co_detection.detect_claimed_role_with_confidence`,
  `public_speech.detect_public_results`, `co_detection.detect_freemason_partner`).

A committed record is re-read under today's readers, so a record an older,
looser reader wrote is found; and a record that reaches the log by any route
that skips the line's own declaration -- a line by another seat, a reader that
stops asking for one -- is found however it got there.

A record that only repeats what already stood -- the same CO by the same seat,
the same verdict by the same claimant -- changes nothing anyone reads. It is
listed (`changed_record` False), not counted.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from app.ai.co_detection import detect_claimed_role_with_confidence, detect_freemason_partner
from app.ai.public_speech import detect_public_results
from app.engine.roles import RoleName
from app.engine.speech_events import CLAIM_CONFIDENCE_THRESHOLD
from app.eval.transcript import GameTranscript, Utterance

_ROLE_EVENTS = frozenset({"role_claim", "role_switch"})
_RESULT_EVENTS = frozenset({"ability_result", "result_correction"})
_PARTNER_EVENT = "partner_claim"
_AUDITED_EVENTS = _ROLE_EVENTS | _RESULT_EVENTS | {_PARTNER_EVENT}
_RESULT_ROLES = {"seer": RoleName.SEER, "medium": RoleName.MEDIUM}


@dataclass(frozen=True)
class UndeclaredRecord:
    """One binding claim event whose own line did not declare it."""

    event_id: str
    event_type: str
    day: int
    actor_id: str
    source_message_id: str
    # The line it was registered from, as the table saw it ("" when the log
    # has no such line) and who posted it.
    text: str
    author_id: str | None
    role: str | None = None
    target_id: str | None = None
    is_werewolf: bool | None = None
    # False when the event only repeated what already stood on the record.
    changed_record: bool = True


def undeclared_records(transcript: GameTranscript) -> tuple[UndeclaredRecord, ...]:
    """Every binding CO, result and partner naming its own line did not declare.

    Empty for a record without a chat log: there is no line to check against.
    """
    state = transcript.final_state or {}
    chat_log = list(state.get("chat_log", []) or [])
    if not chat_log:
        return ()
    messages = {str(message.get("message_id")): message for message in chat_log}
    spoken = _utterance_by_message(transcript.utterances, chat_log)
    found: list[UndeclaredRecord] = []
    standing_role: dict[str, str] = {}
    active_result: dict[tuple[str, str, str], bool] = {}
    partners: set[tuple[str, str]] = set()
    for event in state.get("speech_events", []) or []:
        if float(event.get("confidence", 1.0)) < CLAIM_CONFIDENCE_THRESHOLD:
            continue
        kind = str(event.get("event_type"))
        actor = str(event.get("actor_id"))
        role = event.get("role")
        target = event.get("target_id")
        colour = event.get("result_is_werewolf")
        changed = _changes_record(
            kind, actor, role, target, colour, standing_role, active_result, partners
        )
        if kind in _AUDITED_EVENTS:
            message_id = str(event.get("source_message_id") or "")
            message = messages.get(message_id)
            text = str(message.get("content") or "") if message else ""
            author = str(message.get("author_id")) if message else None
            declared = author == actor and _declared(
                kind,
                actor,
                role,
                target,
                colour,
                text,
                spoken.get(message_id),
                transcript.names,
            )
            if not declared:
                found.append(
                    UndeclaredRecord(
                        event_id=str(event.get("event_id")),
                        event_type=kind,
                        day=int(event.get("day", 0)),
                        actor_id=actor,
                        source_message_id=message_id,
                        text=text,
                        author_id=author,
                        role=str(role) if role else None,
                        target_id=str(target) if target else None,
                        is_werewolf=colour if isinstance(colour, bool) else None,
                        changed_record=changed,
                    )
                )
        _apply(kind, actor, role, target, colour, standing_role, active_result, partners)
    return tuple(found)


def _declared(
    kind: str,
    actor: str,
    role: Any,
    target: Any,
    colour: Any,
    text: str,
    utterance: Utterance | None,
    names: Mapping[str, str],
) -> bool:
    others = {seat: name for seat, name in names.items() if seat != actor}
    spoken_role, confidence = detect_claimed_role_with_confidence(text, list(others.values()))
    binding = spoken_role is not None and confidence >= CLAIM_CONFIDENCE_THRESHOLD
    prose_role = spoken_role.value if binding and spoken_role is not None else None
    if kind in _ROLE_EVENTS:
        if utterance is not None and utterance.public_claim_role == role:
            return True
        return prose_role is not None and prose_role == role
    if kind in _RESULT_EVENTS:
        if utterance is not None and any(
            item.get("target_id") == target
            and bool(item.get("is_werewolf")) == bool(colour)
            and item.get("result_type") in (None, role)
            for item in utterance.public_results
        ):
            return True
        result_role = _RESULT_ROLES.get(str(role))
        if result_role is None:
            return False
        line_claim = utterance.public_claim_role if utterance is not None else None
        claimed_here = line_claim in _RESULT_ROLES or prose_role in _RESULT_ROLES
        return any(
            found.target_id == target and found.is_werewolf == bool(colour)
            for found in detect_public_results(
                text, result_role, others, role_claimed_in_message=claimed_here
            )
        )
    # A partner naming: the line names this seat as the speaker's partner.
    return isinstance(target, str) and detect_freemason_partner(text, others) == target


def _changes_record(
    kind: str,
    actor: str,
    role: Any,
    target: Any,
    colour: Any,
    standing_role: Mapping[str, str],
    active_result: Mapping[tuple[str, str, str], bool],
    partners: set[tuple[str, str]],
) -> bool:
    """Whether the event changes what the record says, or only repeats it."""
    if kind in _ROLE_EVENTS:
        return standing_role.get(actor) != str(role)
    if kind in _RESULT_EVENTS:
        key = (actor, str(role), str(target))
        return key not in active_result or active_result[key] != bool(colour)
    if kind == _PARTNER_EVENT:
        return (actor, str(target)) not in partners
    return True


def _apply(
    kind: str,
    actor: str,
    role: Any,
    target: Any,
    colour: Any,
    standing_role: dict[str, str],
    active_result: dict[tuple[str, str, str], bool],
    partners: set[tuple[str, str]],
) -> None:
    """The record after the event, kept the way `speech_events` derives it."""
    if kind in _ROLE_EVENTS and role:
        standing_role[actor] = str(role)
    elif kind == "role_retraction":
        standing_role.pop(actor, None)
    elif kind in _RESULT_EVENTS and target:
        active_result[(actor, str(role), str(target))] = bool(colour)
    elif kind == "result_retraction" and target:
        active_result.pop((actor, str(role), str(target)), None)
    elif kind == _PARTNER_EVENT and target:
        partners.add((actor, str(target)))


def _utterance_by_message(
    utterances: Sequence[Utterance], chat_log: Sequence[Mapping[str, Any]]
) -> dict[str, Utterance]:
    """Each public line's discussion utterance: the next unclaimed one by the
    same seat, on the same day, with the same text (an utterance is recorded
    before its line is posted, so it does not carry the id)."""
    pool: dict[tuple[str, int, str], list[Utterance]] = defaultdict(list)
    for utterance in utterances:
        if utterance.kind == "discussion":
            pool[(utterance.player_id, utterance.day, utterance.text)].append(utterance)
    found: dict[str, Utterance] = {}
    for message in chat_log:
        if message.get("channel") != "public":
            continue
        key = (
            str(message.get("author_id")),
            int(message.get("day", -1)),
            str(message.get("content") or ""),
        )
        if pool.get(key):
            found[str(message.get("message_id"))] = pool[key].pop(0)
    return found


def describe_record(record: UndeclaredRecord, names: Mapping[str, str]) -> str:
    """The record in words: 「カイト = 共有者CO」, 「アオイ → リク = 黒（占い）」."""
    who = names.get(record.actor_id, record.actor_id)
    if record.event_type in _ROLE_EVENTS:
        return f"{who} = {_ROLE_LABELS.get(record.role or '', record.role or '?')}CO"
    whom = names.get(record.target_id or "", record.target_id or "?")
    if record.event_type == _PARTNER_EVENT:
        return f"{who} の相方 = {whom}"
    colour = "黒" if record.is_werewolf else "白"
    ability = "霊媒" if record.role == "medium" else "占い"
    return f"{who} → {whom} = {colour}（{ability}）"


_ROLE_LABELS = {
    "seer": "占い師",
    "medium": "霊媒師",
    "hunter": "狩人",
    "freemason": "共有者",
}
