"""`transcript-seedN.md` is written in full: nothing a transcript holds is left out.

The record used to print each line and stop: no private notes, no message
ids, no vote evidence, no divination or medium results, no tallies, targets
by seat id. These pin that the record now carries each of those, and that the
committed records of earlier runs still render, every line of them whole.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.eval.report import render_transcript
from app.eval.transcript import DecisionAuditRecord, GameTranscript, Utterance

NAMES = {"p1": "ハルト", "p2": "サクラ", "p3": "レン", "p4": "ソウタ"}
ROLES = {"p1": "seer", "p2": "werewolf", "p3": "hunter", "p4": "fox"}


def _u(kind: str, player: str, day: int, **fields: object) -> Utterance:
    base: dict[str, object] = {
        "day": day,
        "phase": "discussion",
        "kind": kind,
        "player_id": player,
        "player_name": NAMES[player],
        "role": ROLES[player],
        "team": "werewolf" if player == "p2" else ("fox" if player == "p4" else "village"),
        "personality": "冷静",
        "deception_role": None,
    }
    base.update(fields)
    return Utterance(**base)  # type: ignore[arg-type]


def _game() -> GameTranscript:
    long_line = "占い師CO。サクラは黒。" + "理由は" * 80
    utterances = [
        _u("night_action", "p1", 0, phase="night", text="灰を見る", target="p2"),
        _u(
            "discussion",
            "p1",
            1,
            text=long_line,
            stage="immediate",
            reply_to=None,
            directed_question_targets=["p2"],
            public_claim_role="seer",
            public_results=[
                {"result_type": "seer", "target_id": "p2", "is_werewolf": True, "referenced_day": 0}
            ],
            model_text_chars=len(long_line),
            effective_length_limit=120,
            reasoning_memo={
                "overall_thought": "サクラを吊る",
                "execution_target": "p2",
                "suspects": ["p2"],
                "trusted": ["p3"],
                "fox_candidates": ["p4"],
            },
        ),
        _u("discussion", "p2", 1, text="偽の黒だ", stage="rebuttal_or_reassessment", reply_to="m1"),
        _u(
            "discussion",
            "p3",
            1,
            text="ソウタが怪しい。",
            stage="reaction",
            reasoning_memo={"execution_target": "p4"},
        ),
        _u(
            "vote",
            "p1",
            1,
            phase="voting",
            text="黒だから",
            target="p2",
            decision_evidence="0日目夜の黒",
            countercase="占いが偽なら村人",
            alternative_target="p4",
        ),
        _u("vote", "p2", 1, phase="voting", text="偽だから", target="p1"),
        _u("vote", "p3", 1, phase="voting", text="結果を信じる", target="p2"),
        _u("night_action", "p1", 1, phase="night", text="狐を探す", target="p4"),
        _u("night_action", "p3", 1, phase="night", text="占いを守る", target="p1"),
        _u("wolf_chat", "p2", 1, phase="night", text="ソウタを噛む"),
        _u("night_action", "p2", 1, phase="night", text="グレーを削る", target="p4"),
    ]
    final_state = {
        "chat_log": [
            {
                "message_id": "m1",
                "author_id": "p1",
                "content": long_line,
                "channel": "public",
                "day": 1,
            },
            {
                "message_id": "m2",
                "author_id": "p2",
                "content": "偽の黒だ",
                "channel": "public",
                "day": 1,
            },
            {
                "message_id": "m5",
                "author_id": "p3",
                "content": "ソウタが怪しい。",
                "channel": "public",
                "day": 1,
            },
            {
                "message_id": "m3",
                "author_id": "p2",
                "content": "ソウタを噛む",
                "channel": "wolf",
                "day": 1,
            },
        ],
        "divine_records": [
            {"seer_id": "p1", "target_id": "p2", "day": 0, "is_werewolf": True},
            {"seer_id": "p1", "target_id": "p4", "day": 1, "is_werewolf": False},
        ],
        "medium_records": [],
        "guard_records": [{"hunter_id": "p3", "target_id": "p1", "day": 1}],
        "attack_records": [{"wolf_id": "p2", "target_id": "p4", "day": 1, "succeeded": False}],
        "vote_records": [
            {"voter_id": "p1", "target_id": "p2", "day": 1, "round": 1},
            {"voter_id": "p2", "target_id": "p1", "day": 1, "round": 1},
            {"voter_id": "p3", "target_id": "p2", "day": 1, "round": 1},
        ],
        "death_records": [
            {"player_id": "p3", "cause": "first_victim", "day": 0},
            {"player_id": "p2", "cause": "executed", "day": 1},
        ],
        "co_declarations": [],
        "winner": "village",
        "victory_reason": "人狼が全滅",
    }
    audits = [
        DecisionAuditRecord(
            game_id="g",
            day=1,
            phase="voting",
            player_id="p3",
            decision_target="p4",
            displayed_target="p4",
            vote_target="p2",
            public_evidence_ids=(),
            private_evidence_ids=(),
            team_private_evidence_ids=(),
            required_public_result_ids=(),
            published_result_ids=(),
            model_requested_target="p4",
            model_target_was_overridden=False,
            target_change_classification="unexplained",
            target_change_reason="理由が無い",
            vote_round=1,
        )
    ]
    return GameTranscript(
        seed=7,
        names=NAMES,
        roles=ROLES,
        teams={pid: "village" for pid in NAMES},
        personalities={pid: "冷静" for pid in NAMES},
        utterances=utterances,
        final_state=final_state,
        decision_audits=audits,
    )


def test_every_line_is_whole_with_its_id_stage_claims_and_private_note() -> None:
    page = render_transcript(_game())

    line = "占い師CO。サクラは黒。" + "理由は" * 80
    assert f"- `m1` [immediate] **ハルト（占い師／冷静）**: {line}" in page
    assert "名指しの質問: サクラ(p2)" in page
    assert "CO: 占い師" in page
    assert "公開した結果: 占い サクラ(p2)=黒（0日目夜）" in page
    assert "    - 方針: サクラを吊る" in page
    assert "    - 吊り先: サクラ(p2)" in page
    assert "    - 信用: レン(p3)" in page
    assert "    - 妖狐候補: ソウタ(p4)" in page
    assert "`m2` [rebuttal_or_reassessment] **サクラ（人狼／冷静）**: 偽の黒だ" in page
    assert "返信先: m1" in page


def test_votes_carry_their_evidence_and_audit_and_the_day_its_tally() -> None:
    page = render_transcript(_game())

    assert "- **ハルト（占い師／冷静）** → サクラ(p2)" in page
    assert "  - 決め手: 0日目夜の黒" in page
    assert "  - 反対仮説: 占いが偽なら村人" in page
    assert "  - 次点: ソウタ(p4)" in page
    assert "比較の基準（メモの吊り先）: ソウタ(p4)" in page
    # Judged from the record as the analyzer judges it: レン said ソウタ and
    # voted サクラ with nothing new after his line, so the reason is his own.
    assert "判定: 理由のない変更（結果を信じる）" in page
    assert "- 集計（1回目）: サクラ(p2) 2票 / ハルト(p1) 1票" in page
    assert "- 処刑: サクラ(p2)（人狼）" in page


def test_nights_carry_reasons_and_results_and_mornings_their_dead() -> None:
    page = render_transcript(_game())

    assert "- **ハルト（占い師／冷静）** 占い → サクラ(p2)" in page
    assert "  - 結果: 黒（人狼）" in page
    assert "  - 結果: 白（人狼ではない）" in page
    assert (
        "護衛先が襲撃され、守った" not in page
    )  # レン guarded ハルト; the bite went elsewhere
    assert "  - 実際の結果（本人には知らされない）: 護衛先は襲撃されなかった" in page
    # The hunter's and the wolves' outcomes are the god's-eye record, labelled as such;
    # only the seer is told its result.
    assert "  - 実際の結果（本人には知らされない）: ソウタ(p4)への襲撃は失敗" in page
    assert "- 朝の死体: レン(p3)（狩人・初日犠牲者）" in page
    assert "`m3` [人狼チャット] **サクラ（人狼／冷静）**: ソウタを噛む" in page
    assert "- 1日目夜の占い: ハルト(p1) → ソウタ(p4) = 白" in page


_COMMITTED = sorted(
    Path(__file__).resolve().parents[2].glob("artifacts/live-v3-claude-*/**/transcript-seed*.json")
)


@pytest.mark.parametrize("path", _COMMITTED, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_a_committed_record_renders_every_spoken_line_whole(path: Path) -> None:
    transcript = GameTranscript.from_dict(json.loads(path.read_text(encoding="utf-8")))

    page = render_transcript(transcript)

    for utterance in transcript.utterances:
        if utterance.kind in ("discussion", "wolf_chat", "freemason_chat", "vote"):
            flat = " / ".join(part.strip() for part in utterance.text.splitlines() if part.strip())
            assert (flat or utterance.text) in page
    assert page.count("非公開メモ:") == sum(
        1
        for u in transcript.utterances
        if u.kind == "discussion"
        and u.reasoning_memo
        and any(v not in (None, "", [], {}) for v in u.reasoning_memo.values())
    )


def test_there_is_a_committed_record_to_check() -> None:
    assert _COMMITTED
