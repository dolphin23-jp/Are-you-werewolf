"""Which ability results published after a seat's last line explain its ballot (1-D1).

The ballot audit compared every published verdict against a snapshot taken
when the seat spoke. Two faults made that comparison file changes as
"new public evidence" that no new evidence explained:

* the snapshot was taken before the turn's own results were registered, so a
  seer that published in its turn found its own verdicts "new" at the vote;
* any new verdict counted, about anyone, by anyone -- one unrelated result
  explained every change of the day.

Seed 1 of run 13 shows both: ダイキ, a seer claimant, left his memo candidate
ケント for ユイ "as マコトの仮指定", and the audit called it new evidence because
of his own result and アオイ's result on ソウタ. マコト's designation, posted
after ダイキ's last line and naming ユイ, is what the reason cites.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from app.ai.coordinator import AICoordinator
from app.engine.phases import Phase
from app.engine.roles import RoleName
from app.eval.reasoning_analyzer import VoteChangeKind
from app.eval.transcript import GameTranscript, TranscriptRecorder
from tests.ai import first_claude_game as replay
from tests.ai.test_vote_audit import _latest_audit, _name, _PlanAndBallot, _v3_table

SEED1 = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "live-v3-claude-sonnet55-seeds1-3-2026-11-16"
    / "seed1"
    / "transcript-seed1.json"
)


def _spoke_then(publish, reason: str = "気が変わった。"):  # type: ignore[no-untyped-def]
    """The seat names its plan, `publish(controller, state, names, ids)` runs, it votes `other`."""
    controller, state, coordinator, provider, recorder, speaker, plan, other = _v3_table()
    provider.line = f"{_name(state, plan)}が一番怪しい。"
    asyncio.run(coordinator._speak(controller, state, speaker, "initial_view"))
    ids = {"speaker": speaker, "plan": plan, "other": other}
    names = {key: _name(state, value) for key, value in ids.items()}
    publish(controller, state, names, ids)
    provider.vote = other
    provider.reason = reason.format(**names)
    state.phase = Phase.VOTING

    asyncio.run(coordinator._cast_vote(controller, state, speaker))

    return _latest_audit(recorder, speaker, state.day), recorder, ids


def _later_result(
    claimant: str, target: str, line: str = "占いCO。結果を出します。"
):  # type: ignore[no-untyped-def]
    def publish(controller, state, names, ids):  # type: ignore[no-untyped-def]
        author = ids.get(claimant, claimant)
        message_id = controller.chat(author, line.format(**names))
        controller.co(author, RoleName.SEER.value, source_message_id=message_id)
        controller.public_result(
            author, "seer", ids.get(target, target), False, source_message_id=message_id
        )

    return publish


def test_the_seats_own_result_is_not_news_to_it():
    """Registered after the snapshot, the seat's own verdict used to read as new."""

    def own_result(controller, state, names, ids):  # type: ignore[no-untyped-def]
        own_line = state.chat_log[-1]
        assert own_line.author_id == ids["speaker"]
        controller.co(ids["speaker"], RoleName.SEER.value, source_message_id=own_line.message_id)
        controller.public_result(
            ids["speaker"], "seer", "p9", False, source_message_id=own_line.message_id
        )

    audit, _recorder, _ids = _spoke_then(own_result)

    assert audit.target_change_classification == VoteChangeKind.UNEXPLAINED


def test_an_unrelated_result_published_later_explains_nothing():
    audit, _recorder, _ids = _spoke_then(_later_result("p5", "p9"))

    assert audit.target_change_classification == VoteChangeKind.UNEXPLAINED


@pytest.mark.parametrize(
    ("claimant", "target"),
    [("p5", "plan"), ("p5", "other"), ("plan", "p9"), ("other", "p9")],
    ids=["about_old_target", "about_new_target", "by_old_target", "by_new_target"],
)
def test_a_later_result_about_or_by_either_target_is_new_evidence(claimant: str, target: str):
    audit, _recorder, ids = _spoke_then(_later_result(claimant, target))

    assert audit.target_change_classification == VoteChangeKind.NEW_PUBLIC_EVIDENCE


def test_a_reason_that_cites_a_later_speaker_is_a_public_argument_even_with_a_new_result():
    """The seat says who moved it; a verdict published beside the argument does not outrank that."""
    audit, _recorder, _ids = _spoke_then(
        _later_result("p5", "other", "占いCO。{other}を占って白。でも{plan}の方が気になる。"),
        reason="Player5さんの話で{other}に入れる。",
    )

    assert audit.target_change_classification == VoteChangeKind.PUBLIC_ARGUMENT


def test_the_turns_snapshot_includes_the_results_it_published():
    controller, state, coordinator, provider, recorder, speaker, plan, other = _v3_table()
    provider.claim = (RoleName.SEER.value, "p9", False)
    asyncio.run(coordinator._speak(controller, state, speaker, "initial_view"))

    audit = _latest_audit(recorder, speaker, state.day)
    own = [item for item in audit.public_result_ids_at_decision if item.startswith(f"{speaker}:")]
    assert own and all(":p9:" in item for item in own)


def _seed1_state_at_vote(day: int):  # type: ignore[no-untyped-def]
    data = json.loads(SEED1.read_text(encoding="utf-8"))
    return data, replay.replay_at_vote(data, day, session_id="seed1")


def test_seed1_daiki_day1_follows_makotos_later_designation():
    data, state = _seed1_state_at_vote(1)
    transcript = GameTranscript.from_dict(data)
    ids = {name: pid for pid, name in transcript.names.items()}
    daiki, kent, yui = ids["ダイキ"], ids["ケント"], ids["ユイ"]
    reason = next(
        u.text for u in transcript.by_kind("vote") if u.player_id == daiki and u.day == 1
    )
    assert reason.startswith("マコトの仮指定どおりユイに入れる")
    recorder = TranscriptRecorder()
    recorder.transcript = GameTranscript.from_dict(data)
    ai_ids = list(transcript.names)
    coordinator = AICoordinator(
        state, ai_ids, _PlanAndBallot(kent), seed=1, recorder=recorder
    )

    kind, why = coordinator._classify_vote_change(
        state,
        daiki,
        stated_target=kent,
        actual_target=yui,
        candidates=state.votable_ids(daiki),
        reason=reason,
    )

    assert kind == VoteChangeKind.PUBLIC_ARGUMENT
    # m25: マコト's 「ユイさんに仮指定します」, after ダイキ's last line (m3).
    assert "m25" in why
