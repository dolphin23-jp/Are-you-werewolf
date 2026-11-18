"""A CO said again is a restatement, not a new claim (review item 2-A5).

The current-claim view used to return the latest transition, so a seer who
restated its claim on day 2 while giving its morning result moved its CO to
day 2 and to the restating message. Seed 1 of run 13 showed the cost: the day-2
summary listed 「本日のCO: アオイ」 although アオイ had claimed on day 1, and every
later prompt carried it. The claim standing now is the one first made, or the
one switched to; the restatement stays in the history, labelled as such in
the record.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.ai.reasoning import PublicFactLedger, render_public_fact_summary
from app.engine.roles import RoleName
from app.engine.speech_events import (
    RoleClaimStatus,
    SpeechEvent,
    SpeechEventType,
    current_role_claim,
    current_role_claims,
    role_claim_history,
)
from app.eval.report import render_transcript
from app.eval.transcript import GameTranscript
from tests.ai.reasoning.fixtures import declare_co, make_state
from tests.conftest import make_controller

SEED1 = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "live-v3-claude-sonnet55-seeds1-3-2026-11-16"
    / "seed1"
    / "transcript-seed1.json"
)


def _event(
    index: int,
    message_id: str,
    actor: str,
    day: int,
    role: RoleName | None,
    event_type: SpeechEventType = SpeechEventType.ROLE_CLAIM,
) -> SpeechEvent:
    return SpeechEvent(
        event_id=f"e{index}",
        source_message_id=message_id,
        actor_id=actor,
        event_type=event_type,
        day=day,
        role=role,
    )


def test_a_restated_co_keeps_the_day_and_message_it_was_first_made_in():
    controller = make_controller(seed=4)
    controller.state.day = 1
    controller.co("p1", RoleName.SEER.value, source_message_id="m1")

    controller.state.day = 2
    controller.co("p1", RoleName.SEER.value, source_message_id="m8")

    claim = current_role_claim(controller.state.speech_events, "p1")
    assert claim is not None
    assert (claim.day, claim.source_message_id, claim.reaffirmed) == (1, "m1", False)
    assert [(c.day, c.source_message_id) for c in controller.state.co_declarations] == [
        (1, "m1")
    ]
    # The restatement is still on the record, as a restatement.
    history = role_claim_history(controller.state.speech_events, "p1")
    assert [(state.source_message_id, state.reaffirmed) for state in history] == [
        ("m1", False),
        ("m8", True),
    ]


def test_a_switch_is_the_standing_claim_and_moves_to_where_it_was_said():
    events = [
        _event(1, "m1", "p16", 1, RoleName.SEER),
        _event(2, "m3", "p12", 1, RoleName.SEER),
        _event(3, "m9", "p13", 1, RoleName.FREEMASON),
        _event(4, "m20", "p16", 2, RoleName.MEDIUM, SpeechEventType.ROLE_SWITCH),
    ]

    claims = current_role_claims(events)

    assert [(c.player_id, c.role, c.day, c.source_message_id) for c in claims] == [
        ("p12", RoleName.SEER, 1, "m3"),
        ("p13", RoleName.FREEMASON, 1, "m9"),
        ("p16", RoleName.MEDIUM, 2, "m20"),
    ]
    assert claims[-1].status is RoleClaimStatus.SWITCHED


def test_restating_a_switched_claim_keeps_the_switch():
    events = [
        _event(1, "m1", "p1", 1, RoleName.SEER),
        _event(2, "m5", "p1", 2, RoleName.MEDIUM, SpeechEventType.ROLE_SWITCH),
        _event(3, "m9", "p1", 3, RoleName.MEDIUM),
    ]

    claim = current_role_claim(events, "p1")

    assert claim is not None
    assert (claim.role, claim.status, claim.day, claim.source_message_id) == (
        RoleName.MEDIUM,
        RoleClaimStatus.SWITCHED,
        2,
        "m5",
    )


def test_a_restatement_does_not_move_the_claim_in_the_order_of_claims():
    events = [
        _event(1, "m1", "p16", 1, RoleName.SEER),
        _event(2, "m4", "p15", 1, RoleName.SEER),
        _event(3, "m16", "p13", 1, RoleName.FREEMASON),
        _event(4, "m34", "p15", 2, RoleName.SEER),
    ]

    claims = current_role_claims(events)

    assert [(c.player_id, c.day, c.source_message_id) for c in claims] == [
        ("p16", 1, "m1"),
        ("p15", 1, "m4"),
        ("p13", 1, "m16"),
    ]


def test_the_days_summary_lists_a_switch_but_not_a_restatement():
    state = make_state(day=2, roles={"p5": RoleName.SEER})
    declare_co(state, "p5", RoleName.SEER, day=1, source_message_id="m1")
    declare_co(state, "p6", RoleName.SEER, day=1, source_message_id="m2")
    declare_co(state, "p5", RoleName.SEER, day=2, source_message_id="m8")
    declare_co(state, "p6", RoleName.MEDIUM, day=2, source_message_id="m9")

    summary = render_public_fact_summary(PublicFactLedger(state), 2)

    todays = [line for line in summary.splitlines() if line.startswith("- 本日のCO")]
    assert todays == ["- 本日のCO: Player6(p6)=霊媒師"]


def _seed1() -> GameTranscript:
    return GameTranscript.from_dict(json.loads(SEED1.read_text(encoding="utf-8")))


def test_seed_1_aoi_s_day_2_restatement_leaves_her_claim_on_day_1():
    """アオイ claimed seer in m4 on day 1 and said it again with her result in m34."""
    transcript = _seed1()
    events = [
        SpeechEvent(
            **{
                **raw,
                "event_type": SpeechEventType(raw["event_type"]),
                "role": RoleName(raw["role"]) if raw["role"] else None,
            }
        )
        for raw in transcript.final_state["speech_events"]
    ]

    claim = current_role_claim(events, "p15")

    assert claim is not None
    assert (claim.day, claim.source_message_id) == (1, "m4")
    # Only アカリ's medium CO is new on day 2.
    assert [c.player_id for c in current_role_claims(events) if c.day == 2] == ["p1"]


def test_the_record_labels_a_restatement_as_one():
    page = render_transcript(_seed1())

    m34 = next(line for line in page.splitlines() if line.startswith("- `m34` "))
    detail = page.splitlines()[page.splitlines().index(m34) + 1]
    assert "CO（再表明）: 占い師" in detail
    m4 = next(line for line in page.splitlines() if line.startswith("- `m4` "))
    first = page.splitlines()[page.splitlines().index(m4) + 1]
    assert "CO: 占い師" in first and "再表明" not in first
    # The system record lists the claim on the day it was made, and the
    # restatement as a restatement -- not as a day-2 CO.
    assert "- 1日目のCO: アオイ(p15) = 占い師（m4）" in page
    assert "- 2日目のCO（再表明）: アオイ(p15) = 占い師（m34）" in page
    assert "- 2日目のCO: アオイ(p15)" not in page
