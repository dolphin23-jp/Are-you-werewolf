"""A CO or a result nobody declared never reaches the public record (X22).

Run 16, day 1, m9: カイト, a wolf told to lurk, said 「共有CO二人は了解」 -- he
acknowledged the two freemason claims. His turn's `public_claim_role` was
empty, so the free-text reader in `claims.build_claim_drafts` read the prose,
matched 共有 + CO, and registered a binding freemason CO (e10, confidence
0.9). The board memo then listed 「共有=マコト、ホノカ、カイト」 for the rest of
the game, the CO panel showed it to the human, and カイト took up the claim the
record gave him.

The same layer put run 15's 「リク黒が外れたのはダイキさんの問題」 on the record
as アオイ's own black on リク (X13). The survey of every committed v3 record
found two more lines it registered a CO from without a declaration (both
restatements of a claim the seat already held). Each is replayed here on the
board as it stood just before the line, through the same drafting the game
uses, and must register nothing now.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.ai.reasoning import PublicFactLedger
from app.ai.reasoning.claims import build_claim_drafts, register_claim_drafts
from app.ai.schemas import DiscussionOutput, PublicResultClaim
from app.engine.phases import Phase
from app.engine.roles import RoleName
from app.engine.speech_events import CLAIM_CONFIDENCE_THRESHOLD, SpeechEventType
from app.main import app
from app.sessions.store import get_session_store
from tests.ai.reasoning.record_boards import board_from_record
from tests.conftest import make_controller

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
RUN16 = ARTIFACTS / "live-v3-claude-sonnet55-seed1-2026-11-18b" / "transcript-seed1.json"
M9 = (
    "共有CO二人は了解。占いは2-1だから、今日はグレランで灰の中身を見たい。"
    "薄い発言から順に拾っていくよ。"
)

# (record, message, speaker, what the old reader put on the record from it)
SURVEY = [
    # run 16 day 1: the third freemason
    ("live-v3-claude-sonnet55-seed1-2026-11-18b/transcript-seed1.json", "m9", "p8", "freemason"),
    # run 15 day 5 (X13): ダイキ's black on リク, as アオイ's own
    ("live-v3-claude-sonnet55-seed1-2026-11-18/transcript-seed1.json", "m115", "p15", "p10:black"),
    # run 15 day 4: 「共有だから私は白で合ってるよ」, a restatement read from a reason
    ("live-v3-claude-sonnet55-seed1-2026-11-18/transcript-seed1.json", "m103", "p13", "freemason"),
    # run 13 seed 2 day 2: 「霊媒COするって決めてたよ」, a restatement read from a story
    (
        "live-v3-claude-sonnet55-seeds1-3-2026-11-16/seed2/transcript-seed2.json",
        "m32",
        "p16",
        "medium",
    ),
]

_BINDING_TYPES = (
    SpeechEventType.ROLE_CLAIM,
    SpeechEventType.ROLE_SWITCH,
    SpeechEventType.ABILITY_RESULT,
)


def _load(path: str) -> dict[str, Any]:
    return json.loads((ARTIFACTS / path).read_text(encoding="utf-8"))


def _line(record: dict[str, Any], message_id: str) -> tuple[dict[str, Any], DiscussionOutput, str]:
    """The message, the turn's output as the model returned it, and the board cut before it."""
    final = record["final_state"]
    chat = final["chat_log"]
    index = next(i for i, m in enumerate(chat) if m["message_id"] == message_id)
    message = chat[index]
    previous = [m for m in chat[:index] if m["day"] == message["day"]]
    at = previous[-1]["message_id"] if previous else "morning"
    utterance = next(
        u
        for u in record["utterances"]
        if u["kind"] == "discussion"
        and u["player_id"] == message["author_id"]
        and u["day"] == message["day"]
        and u["text"] == message["content"]
    )
    output = DiscussionOutput(
        public_message=message["content"],
        public_claim_role=utterance["public_claim_role"],
        public_results=[PublicResultClaim(**item) for item in utterance["public_results"]],
    )
    return message, output, at


def _old_record(record: dict[str, Any], message_id: str) -> list[str]:
    found = []
    for event in record["final_state"]["speech_events"]:
        if event["source_message_id"] != message_id or event["confidence"] < 0.8:
            continue
        if event["event_type"] == "role_claim":
            found.append(event["role"])
        elif event["event_type"] == "ability_result":
            colour = "black" if event["result_is_werewolf"] else "white"
            found.append(f"{event['target_id']}:{colour}")
    return found


@pytest.mark.parametrize(("path", "message_id", "speaker", "registered"), SURVEY)
def test_each_line_the_survey_found_registers_nothing_now(
    path: str, message_id: str, speaker: str, registered: str
):
    record = _load(path)
    message, output, at = _line(record, message_id)
    assert message["author_id"] == speaker
    # The structured output declared nothing: the record came from the prose.
    assert output.public_claim_role is None and output.public_results == []
    assert registered in _old_record(record, message_id)

    board = board_from_record(record["final_state"], "replay", message["day"], at)
    drafts = build_claim_drafts(
        output, PublicFactLedger(board), speaker_id=speaker, names_only=True
    )

    assert [
        draft
        for draft in drafts
        if draft.event_type in _BINDING_TYPES and draft.confidence >= CLAIM_CONFIDENCE_THRESHOLD
    ] == []


def test_run16_m9_leaves_the_record_the_memo_and_the_players_view_alone():
    """X22-2: the human's CO panel reads `co_declarations`, the record's own view."""
    record = json.loads(RUN16.read_text(encoding="utf-8"))
    controller = make_controller()
    controller.state = board_from_record(record["final_state"], "run16", 1, "m8")
    state = controller.state
    assert [(c.player_id, c.claimed_role) for c in state.co_declarations if c.claimed_role
            is RoleName.FREEMASON] == [("p0", RoleName.FREEMASON), ("p13", RoleName.FREEMASON)]

    output = DiscussionOutput(public_message=M9)
    drafts = build_claim_drafts(output, PublicFactLedger(state), speaker_id="p8", names_only=True)
    register_claim_drafts(controller, "p8", drafts, "m9")

    assert all(claim.player_id != "p8" for claim in state.co_declarations)
    assert all(event.actor_id != "p8" for event in state.speech_events)
    view = state.get_player_view("p5")
    shown = {c["player_id"] for c in view["co_declarations"] if c["claimed_role"] == "freemason"}
    assert shown == {"p0", "p13"}
    # The board memo every seat reads the CO list from follows the record.
    ledger = PublicFactLedger(state)
    claimed = {claim.player_id for claim in ledger.co_declarations()}
    assert claimed == {"p16", "p15", "p0", "p1", "p13"}


def test_a_human_acknowledging_the_freemason_claims_is_not_shown_as_one():
    """The human path: the same reader, the same record, the same panel."""
    client = TestClient(app)
    response = client.post("/api/games", json={"human_name": "了解する人", "seed": 15})
    session_id = response.json()["session_id"]
    human_id = response.json()["human_player_id"]
    session = get_session_store().get(session_id)
    assert session is not None
    session.controller.state.phase = Phase.DISCUSSION

    response = client.post(f"/api/games/{session_id}/chat", json={"content": M9})

    assert response.status_code == 200
    view = client.get(f"/api/games/{session_id}/view").json()
    assert all(claim["player_id"] != human_id for claim in view["co_declarations"])
